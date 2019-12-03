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
    model, _ = utils.load_ensemble_for_inference(
        path.split(':'), task, model_arg_overrides=None,
    )
    assert len(model) == 1
    real_teacher_model = model[0].to('cuda:1')
    real_teacher_model.eval()
    self.teacher_model = [real_teacher_model]  # avoid being count as param
    self.alpha_ce = args.alpha_ce
    self.alpha_clm = args.alpha_clm
    self.alpha_cos = args.alpha_cos
    self.temp = args.distill_temp

  @staticmethod
  def add_args(parser):
    parser.add_argument('--teacher-model', type=str, metavar='PATH',
                        help='storage location of teacher model')
    parser.add_argument('--alpha-ce', default=1.0, type=float,
                        help='distill loss weight for distillation loss')
    parser.add_argument('--alpha-clm', default=1.0, type=float,
                        help='distill loss weight for language model loss')
    parser.add_argument('--alpha-cos', default=3.0, type=float,
                        help='distill loss weight for cosine similarity loss')
    parser.add_argument('--distill-temp', default=1.0, type=float,
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

    seq_len = sample['net_input']['src_tokens'].size(1)
    block_size = seq_len // 4
    teacher_output, _ = self.teacher_model[0](sample['net_input']['src_tokens'].to('cuda:1'), None)
    teacher_output = teacher_output / self.temp
    ce_loss = 0.0
    for block_start in range(0, seq_len - 1, block_size):
        teacher_dist = self.teacher_model[0].get_normalized_probs([teacher_output[:, block_start:block_start+block_size, :]], log_probs=False, no_backward=True)
        lprobs = model.get_normalized_probs(net_output[:, block_start:block_start+block_size, :], log_probs=True)
        ce_loss -= torch.sum(lprobs * teacher_dist.to('cuda:0'))

    remainder_size = seq_len % block_size
    teacher_dist = self.teacher_model[0].get_normalized_probs([teacher_output[:, -remainder_size:, :]], log_probs=False, no_backward=True)
    lprobs = model.get_normalized_probs(net_output[:, -remainder_size:, :], log_probs=True)
    ce_loss -= torch.sum(lprobs * teacher_dist.to('cuda:0'))

    target = net_output[0].new(net_output[0].size(0)).fill_(1)
    cos_loss += F.cosine_embedding_loss(net_output[0], teacher_output.to('cuda:0'), target)

    loss = self.alpha_ce * ce_loss + self.alpha_clm * clm_loss + self.alpha_cos * cos_loss

    orig = utils.strip_pad(orig_target, self.padding_idx)
    ntokens = orig.numel()
    sample_size = sample['target'].size(0) if self.args.sentence_avg else ntokens
    logging_output = {
        'loss': utils.item(loss.data) if reduce else loss.data,
        'ce': ce_loss.data.item(),
        'clm': clm_loss.data.item(),
        'cos': cos_loss.data.item(),
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
    cos_sum = sum(log.get('cos', 0) for log in logging_outputs)
    ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
    nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
    sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
    agg_output = {
        'loss': loss_sum / sample_size / math.log(2),
        'nll_loss': loss_sum / sample_size / math.log(2),
        'ce': ce_sum / sample_size / math.log(2),
        'clm': clm_sum / sample_size / math.log(2),
        'cos': cos_sum / sample_size / math.log(2),
        'ntokens': ntokens,
        'nsentences': nsentences,
        'sample_size': sample_size,
    }
    if sample_size != ntokens:
        agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
    return agg_output
