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


def label_smoothed_nll_loss_with_rej_cortes_etal(lprobs, rej_probs, target, epsilon, c, ignore_index=None, reduce=True, burn_in=False):
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


def label_smoothed_nll_loss_with_rej_geifman_etal(lprobs, rej_probs, target, epsilon, c,
        aux_lprobs=None, ignore_index=None, reduce=True, burn_in=False, lambda_=32):

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

        if aux_lprobs is not None:
            aux_nlprobs = -aux_lprobs.gather(dim=-1, index=target)
            aux_nlprobs = aux_nlprobs[non_pad_mask]
        else:
            aux_nlprobs = None

        reward = torch.sum(nll_loss * (1 - rej_probs)) / (1 - rej_probs).sum() * target.numel()  # we don't want to normalize implicitly
        constraint = torch.clamp(rej_probs.sum() - c * target.numel(), min=0.0) ** 2  # reduce needs to happen here -- we want to regularize the mean, not each item

        nll_loss = reward + lambda_ * constraint
    else:
        rej_probs = torch.zeros_like(nll_loss)
        aux_nlprobs = None

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        rej_probs = rej_probs.sum()
        aux_nlprobs = aux_nlprobs.sum() if aux_nlprobs is not None else None
    else:
        raise NotImplementedError("Geifman loss requires sampling to estimate the rejection reward, so sampling is enforced")

    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    if not burn_in and aux_nlprobs is not None:
        loss += aux_nlprobs

    return loss, nll_loss, rej_probs, aux_nlprobs


@register_criterion('label_smoothed_cross_entropy_with_rej')
class LabelSmoothedCrossEntropyWithRejectionCriterion(FairseqCriterion):

    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        rejection_formulation="cortes",
        rejection_margin=0.005,
        rejection_burn_in_updates=4000,
        rejection_ratio=0.001,
        rejection_lambda=32,
        rejection_option_auxiliary_head=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.loss_fml = rejection_formulation
        self.xi = rejection_margin
        self.burn_in_updates = rejection_burn_in_updates
        self.c = rejection_ratio
        self.lambda_ = rejection_lambda
        self.use_auxiliary_head = rejection_option_auxiliary_head

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--rejection-formulation', choices=["cortes", "geifman"], default="cortes",
                            help='the version of rejection option formulation to use')
        parser.add_argument('--rejection-margin', type=float, metavar='Xi', default=0.005,
                            help='use rejection option during training')
        parser.add_argument('--rejection-burn-in-updates', type=int, metavar='N', default=4000,
                            help='let hypothesis model burn-in for N steps before starting to train rejection head')
        parser.add_argument('--rejection-ratio', type=float, metavar='c', default=0.001,
                            help='target ratio of training data to be rejected')
        parser.add_argument('--rejection-lambda', type=float, metavar='lambda', default=32,
                            help='rejection ratio constraint strength')
        # fmt: on

    def forward(self, model, sample, reduce=True, update_no=-1):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, rej_probs, aux_nlprobs = self.compute_loss(model, net_output, sample, reduce=reduce, update_no=update_no)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'rej_probs': utils.item(rej_probs.data),
            'aux_nlprobs': utils.item(aux_nlprobs.data),
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, update_no=-1):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        rej_probs = model.get_rejection_probs(net_output, log_probs=False)
        if self.use_auxiliary_head:
            aux_lprobs = model.get_auxiliary_probs(net_output, log_probs=True)
            aux_lprobs = aux_lprobs.view(-1, aux_lprobs.size(-1))
        else:
            aux_lprobs = None
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        thres = 1.0 / model.decoder.embed_tokens.num_embeddings + self.xi
        if self.loss_fml == "cortes":
            loss, nll_loss, rej_probs = label_smoothed_nll_loss_with_rej_cortes_etal(
                lprobs, rej_probs, target, self.eps, thres, ignore_index=self.padding_idx, reduce=reduce,
                burn_in=(update_no < self.burn_in_updates)
            )
            aux_nlprobs = torch.zeros(1)
        elif self.loss_fml == "geifman":
            loss, nll_loss, rej_probs, aux_nlprobs = label_smoothed_nll_loss_with_rej_geifman_etal(
                lprobs, rej_probs, target, self.eps, self.c, aux_lprobs=aux_lprobs,
                ignore_index=self.padding_idx, reduce=reduce,
                burn_in=(update_no < self.burn_in_updates),
                lambda_=self.lambda_,
            )
            aux_nlprobs = torch.zeros(1) if aux_nlprobs is None else aux_nlprobs
        return loss, nll_loss, rej_probs, aux_nlprobs

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
            'aux_loss': sum(log.get('aux_nlprobs', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
