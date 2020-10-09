# -*- coding: utf-8 -*-
#
# Copyright © 2020 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2020-09-16
#
# Distributed under terms of the MIT license.

import math
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion


def label_smoothed_nll_loss_with_rej(lprobs, rej_probs, target, epsilon, c, ignore_index=None, reduce=True, burn_in=False):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if not burn_in:
        rej_probs = rej_probs.view(-1, rej_probs.size(2))
        rej_probs = rej_probs[non_pad_mask].squeeze(-1)
        nll_loss = (1 - rej_probs) * nll_loss - rej_probs * math.log(c)
    else:
        rej_probs = torch.zeros_like(nll_loss)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        rej_probs = rej_probs.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss, rej_probs


@register_criterion('label_smoothed_cross_entropy_with_rej')
class LabelSmoothedCrossEntropyWithRejectionCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.xi = args.rejection_margin
        self.burn_in_updates = args.rejection_burn_in_updates

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--rejection-margin', type=float, metavar='Xi', default=0.005,
                            help='use rejection option during training')
        parser.add_argument('--rejection-burn-in-updates', type=int, metavar='N', default=4000,
                            help='let hypothesis model burn-in for N steps before starting to train rejection head')
        # fmt: on

    def forward(self, model, sample, reduce=True, update_no=-1):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, rej_probs = self.compute_loss(model, net_output, sample, reduce=reduce, update_no=update_no)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'rej_probs': rej_probs,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, update_no=-1):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        rej_probs = model.get_rejection_probs(net_output, log_probs=False)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        c = 1.0 / model.decoder.embed_tokens.num_embeddings + self.xi
        loss, nll_loss, rej_probs = label_smoothed_nll_loss_with_rej(
            lprobs, rej_probs, target, self.eps, c, ignore_index=self.padding_idx, reduce=reduce,
            burn_in=(update_no < self.burn_in_updates)
        )
        return loss, nll_loss, rej_probs

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'rej_probs': sum(log.get('rej_probs', 0) for log in logging_outputs) / ntokens if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
