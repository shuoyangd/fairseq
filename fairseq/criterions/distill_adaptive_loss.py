# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import math
import pdb
import torch
import torch.nn.functional as F

from fairseq import utils
from . import FairseqCriterion, register_criterion
from .adaptive_loss import AdaptiveLoss

@register_criterion('distill_adaptive_loss')
class DistillAdaptiveLoss(AdaptiveLoss):

  def __init__(self, args, task):
    super().__init__(args, task)
    path = args.teacher_model
    model, model_args = utils.load_ensemble_for_inference(
        path.split(':'), task, model_arg_overrides=None,
    )
    assert len(model) == 1
    real_teacher_model = model[0].cuda()
    real_teacher_model.eval()
    self.teacher_model = [real_teacher_model]  # avoid being count as param
    self.alpha = args.distill_alpha
    self.temp = args.distill_temp

  @staticmethod
  def add_args(parser):
    parser.add_argument('--teacher-model', type=str, metavar='PATH',
                        help='storage location of teacher model')
    parser.add_argument('--distill-alpha', default=1.0, type=float,
                        help='distill loss weight')
    parser.add_argument('--distill-temp', default=8.0, type=float,
                        help='distill teacher distribution temperature')

  def forward(self, model, sample, reduce=True):
    """Compute the loss for the given sample.

    Returns a tuple with three elements:
    1) the loss
    2) the sample size, which is used as the denominator for the gradient
    3) logging outputs to display while training
    """

    assert hasattr(model.decoder, 'adaptive_softmax') and model.decoder.adaptive_softmax is not None
    adaptive_softmax = model.decoder.adaptive_softmax

    net_output = model(**sample['net_input'])
    orig_target = model.get_targets(sample, net_output)

    nsentences = orig_target.size(0)
    orig_target = orig_target.view(-1)

    bsz = orig_target.size(0)

    logits, target = adaptive_softmax(net_output[0], orig_target)
    assert len(target) == len(logits)

    clm_loss = net_output[0].new(1 if reduce else bsz).zero_()

    for i in range(len(target)):
        if target[i] is not None:
            assert (target[i].min() >= 0 and target[i].max() <= logits[i].size(1))
            clm_loss += F.cross_entropy(logits[i], target[i], size_average=False, ignore_index=self.padding_idx,
                                    reduce=reduce)

    teacher_output, _ = self.teacher_model[0](sample['net_input']['src_tokens'], None)
    teacher_output = teacher_output / self.temp
    teacher_dist = self.teacher_model[0].get_normalized_probs([teacher_output], log_probs=False, no_backward=True).detach()
    lprobs = model.get_normalized_probs(net_output, log_probs=True)
    ce_loss = -torch.sum(lprobs * teacher_dist)

    loss = self.alpha * ce_loss + (1 - self.alpha) * clm_loss

    orig = utils.strip_pad(orig_target, self.padding_idx)
    ntokens = orig.numel()
    sample_size = sample['target'].size(0) if self.args.sentence_avg else ntokens
    logging_output = {
        'loss': utils.item(loss.data) if reduce else loss.data,
        'ce': ce_loss.data.item(),
        'clm': clm_loss.data.item(),
        'ntokens': ntokens,
        'nsentences': nsentences,
        'sample_size': sample_size,
    }
    return loss, sample_size, logging_output

  @staticmethod
  def aggregate_logging_outputs(logging_outputs):
    """Aggregate logging outputs from data parallel training."""
    loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
    ce_sum = sum(log.get('ce', 0) for log in logging_outputs)
    clm_sum = sum(log.get('clm', 0) for log in logging_outputs)
    ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
    nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
    sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
    agg_output = {
        'loss': loss_sum / sample_size / math.log(2),
        'nll_loss': loss_sum / sample_size / math.log(2),
        'ce': ce_sum / sample_size / math.log(2),
        'clm': clm_sum / sample_size / math.log(2),
        'ntokens': ntokens,
        'nsentences': nsentences,
        'sample_size': sample_size,
    }
    if sample_size != ntokens:
        agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
    return agg_output
