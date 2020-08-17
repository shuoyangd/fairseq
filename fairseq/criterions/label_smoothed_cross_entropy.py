# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import math
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion

HDF5_CHUNK_SIZE = 1024

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
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
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

        self.decoder_states_dump_dir = args.decoder_states_dump_dir
        self.dump_ok_bad_label = args.dump_ok_bad_label
        if self.decoder_states_dump_dir is not None:
            self.decoder_states_dump_file = h5py.File(self.decoder_states_dump_dir, 'w')
            self.state_tokens_dump_file = h5py.File(self.decoder_states_dump_dir + ".tokens", 'w')
            if self.dump_ok_bad_label:
                self.ok_bad_label_file = open(self.decoder_states_dump_dir + ".labels", 'w')
            self.dump_log_file = open(self.decoder_states_dump_dir + ".log", 'w')
            self.not_dummy_batch = False
            self.decoder_states_dataset = \
                self.decoder_states_dump_file.create_dataset("decoder_states",
                                                             (0, args.decoder_embed_dim),
                                                             maxshape=(None, None),
                                                             dtype='f',
                                                             chunks=(HDF5_CHUNK_SIZE, args.decoder_embed_dim))
            self.state_tokens_dataset = \
                self.state_tokens_dump_file.create_dataset("tokens",
                                                           (0,),
                                                           maxshape=(None,),
                                                           dtype='i',
                                                           chunks=(HDF5_CHUNK_SIZE,))
            self.decoder_states_count = 0
        else:
            self.decoder_states_dump_file = None

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    @staticmethod
    def write_ok_bad_labels(f, labels):
        for label in labels:
            if label.item() == 0:
                f.write("BAD\n")
            else:
                f.write("OK\n")

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        if self.decoder_states_dump_dir is not None and self.not_dummy_batch:

            x = net_output[0].detach().contiguous().view(-1, net_output[0].size(-1))
            gold = model.get_targets(sample, net_output).view(-1)
            pad_filter = (gold != self.padding_idx)
            x = x[pad_filter, :]
            gold = gold[pad_filter]

            # another option: use the argmax for class assignment
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            _, target = torch.max(lprobs, dim=-1)
            target = target[pad_filter]
            ok_bad_labels = (gold == target)

            remaining_dp_counts = x.size(0)
            initial_chunk_size = min(remaining_dp_counts, HDF5_CHUNK_SIZE - (self.decoder_states_count - 1) % HDF5_CHUNK_SIZE - 1)
            self.decoder_states_dataset[self.decoder_states_count:self.decoder_states_count+initial_chunk_size, :] = x[:initial_chunk_size, :].cpu()
            self.state_tokens_dataset[self.decoder_states_count:self.decoder_states_count+initial_chunk_size] = target[:initial_chunk_size].cpu()
            if self.dump_ok_bad_label:
                LabelSmoothedCrossEntropyCriterion.write_ok_bad_labels(self.ok_bad_label_file, ok_bad_labels[:initial_chunk_size])
            self.decoder_states_count += initial_chunk_size
            remaining_dp_counts -= initial_chunk_size
            while remaining_dp_counts > 0:
                self.decoder_states_dataset.resize(self.decoder_states_dataset.shape[0] + HDF5_CHUNK_SIZE, axis=0)
                self.state_tokens_dataset.resize(self.state_tokens_dataset.shape[0] + HDF5_CHUNK_SIZE, axis=0)
                chunk_size = min(remaining_dp_counts, HDF5_CHUNK_SIZE)
                startpoint = x.size(0) - remaining_dp_counts
                self.decoder_states_dataset[self.decoder_states_count:self.decoder_states_count+chunk_size, :] =\
                    x[startpoint:startpoint+chunk_size, :].cpu()
                self.state_tokens_dataset[self.decoder_states_count:self.decoder_states_count+chunk_size] =\
                    target[startpoint:startpoint+chunk_size].cpu()
                if self.dump_ok_bad_label:
                    LabelSmoothedCrossEntropyCriterion.write_ok_bad_labels(self.ok_bad_label_file, ok_bad_labels[startpoint:startpoint+chunk_size])
                self.decoder_states_count += chunk_size
                remaining_dp_counts -= chunk_size
            self.dump_log_file.write("current batch has {0} states, writing into a database with capacity {1}. after writing, there are {2} states in database\n".format(x.size(0), self.decoder_states_dataset.shape[0], self.decoder_states_count))

        self.not_dummy_batch = True
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
