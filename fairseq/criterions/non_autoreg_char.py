# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn.functional as F

import pdb

from fairseq import utils

from . import FairseqCriterion, register_criterion
from fairseq.models import NonAutoRegCharGenerator, LengthPredictionGenerator, RefinementAutoEncoder


@register_criterion('non_autoreg')
class NonAutoRegCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        num_refinements = len(lprobs)
        if num_refinements == 1:
            lprobs = lprobs[0]
        elif num_refinements > 1:
            from functools import reduce
            lprobs = reduce(lambda x, y: x + y, lprobs)
        else:
            raise ValueError("no log probability calculated")

        target = model.get_targets(sample, net_output)

        assert hasattr(model, "generator") and \
            isinstance(model.generator, NonAutoRegCharGenerator)

        # FIXME: sample
        """
        lprobs_pred = lprobs.clone()
        lprobs_pred[:, :, :, self.padding_idx] = -99
        _, idxes = torch.max(lprobs_pred, dim=-1)
        sys.stdout.buffer.write(("pred: " + self.tgt_dict.string(idxes[0, 0]) + "\n").encode('utf-8'))
        sys.stdout.buffer.write(("tgt: " + self.tgt_dict.string(target[0, 0]) + "\n").encode('utf-8'))
        """

        # length prediction loss
        lp_loss = torch.sum(torch.cuda.FloatTensor([0.0]))
        max_word_len = model.generator.max_word_len
        if not model.generator.length_predictor:
            # if we are making length predictions,
            # pad/truncate the target to be of the same length as max_word_len
            # (we don't operate on predictions because there is no good way to pad the predictions)
            if target.size(2) < max_word_len:
                pad = torch.ones_like(lprobs[:, :, :max_word_len - target.size(2), 0]).long()  # lprobs only provides the shape
                target = torch.cat([target, pad], dim=2)
            elif target.size(2) > max_word_len:
                target = target[:, :, :max_word_len]
        else:
            length_dist = model.generator.length_predictor(net_output[0])
            valid_token = (target != self.tgt_dict.pad_index)
            length_target = torch.sum(valid_token, dim=-1).long()  # (batch_size, max_seq_len)
            length_target[length_target > max_word_len - 1] = max_word_len - 1

            length_dist = length_dist.view(-1, length_dist.size(-1))
            length_target = length_target.contiguous().view(-1)
            lp_loss = F.nll_loss(length_dist, length_target, size_average=False, reduce=reduce)

            if lprobs.size(2) > target.size(2):
                lprobs = lprobs[:, :, :target.size(2), :]
            elif lprobs.size(2) < target.size(2):
                target = target[:, :, :lprobs.size(2)]

        dae_loss = torch.sum(torch.cuda.FloatTensor([0.0]))
        if model.generator.refinement_autoenc:
            vocab_size = len(self.tgt_dict)
            corrupted_target = utils.corrupt_process(target, vocab_size)
            dae_dist = model.generator.refinement_autoenc(corrupted_target, net_output[0])
            dae_dist = dae_dist.view(-1, dae_dist.size(-1))
            flat_target = target.contiguous().view(-1)
            dae_loss = F.nll_loss(dae_dist, flat_target, size_average=False, ignore_index=self.padding_idx,
                                  reduce=reduce)
        dae_loss *= num_refinements

        # MT loss
        flat_lprobs = lprobs.contiguous().view(-1, lprobs.size(-1))
        flat_target = target.contiguous().view(-1)
        assert flat_lprobs.size(0) == flat_target.size(0)
        mt_loss = F.nll_loss(flat_lprobs, flat_target, size_average=False, ignore_index=self.padding_idx,
                          reduce=reduce)

        loss = mt_loss + lp_loss + dae_loss
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'mt_loss': utils.item(mt_loss.data) if reduce else mt_loss.data,
            'lp_loss': utils.item(lp_loss.data) if reduce else lp_loss.data,
            'dae_loss': utils.item(dae_loss.data) if reduce else dae_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get('mt_loss', 0) for log in logging_outputs)
        lp_loss_sum = sum(log.get('lp_loss', 0) for log in logging_outputs)
        dae_loss_sum = sum(log.get('dae_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'mt_loss': mt_loss_sum / sample_size / math.log(2),
            'lp_loss': lp_loss_sum / sample_size / math.log(2),
            'dae_loss': dae_loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = mt_loss_sum / ntokens / math.log(2)
        return agg_output
