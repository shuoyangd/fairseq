# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import pdb
import sys

import torch
from fairseq import utils
from fairseq.criterions.label_smoothed_cross_entropy import HDF5_CHUNK_SIZE

DECODER_EMBED_DIM = 1024

class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(
        self,
        tgt_dict,
        softmax_batch=None,
        compute_alignment=False,
        eos=None,
        symbols_to_strip_from_output=None,
        decoder_states_dump_dir=None,
    ):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )

        self.decoder_states_dump_dir = decoder_states_dump_dir
        if self.decoder_states_dump_dir is not None:
            self.decoder_states_dump_file = h5py.File(self.decoder_states_dump_dir, 'w')
            self.dump_log_file = open(self.decoder_states_dump_dir + ".log", 'w')
            self.decoder_states_dataset = \
                self.decoder_states_dump_file.create_dataset("decoder_states",
                                                             (0, DECODER_EMBED_DIM),
                                                             maxshape=(None, None),
                                                             dtype='f',
                                                             chunks=(HDF5_CHUNK_SIZE, DECODER_EMBED_DIM))
            self.decoder_states_count = 0
            self.decoder_probs_dump_file = open(self.decoder_states_dump_dir + ".probs", 'w')
            self.ok_bad_tags_dump_file = open(self.decoder_states_dump_dir + ".tags", 'w')
        else:
            self.decoder_states_dump_file = None

    @torch.no_grad()
    def generate(self, models, sample, return_states=False, **kwargs):
        """Score a batch of translations."""
        net_input = sample["net_input"]

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample["target"]

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1] if len(decoder_out) > 1 else None
            if type(attn) is dict:
                attn = attn.get("attn", None)

            x = decoder_out[0].detach().contiguous().view(-1, decoder_out[0].size(-1))
            if self.decoder_states_dump_dir is not None:
                gold = sample['target']
                pad_filter = (gold != self.pad)
                x = x[pad_filter.view(-1), :]

                remaining_dp_counts = x.size(0)
                initial_chunk_size = min(remaining_dp_counts, HDF5_CHUNK_SIZE - (self.decoder_states_count - 1) % HDF5_CHUNK_SIZE - 1)
                self.decoder_states_dataset[self.decoder_states_count:self.decoder_states_count+initial_chunk_size, :] = \
                        x[:initial_chunk_size, :].cpu()
                self.decoder_states_count += initial_chunk_size
                remaining_dp_counts -= initial_chunk_size
                while remaining_dp_counts > 0:
                    self.decoder_states_dataset.resize(self.decoder_states_dataset.shape[0] + HDF5_CHUNK_SIZE, axis=0)
                    chunk_size = min(remaining_dp_counts, HDF5_CHUNK_SIZE)
                    startpoint = x.size(0) - remaining_dp_counts
                    self.decoder_states_dataset[self.decoder_states_count:self.decoder_states_count+chunk_size, :] =\
                        x[startpoint:startpoint+chunk_size, :].cpu()
                    self.decoder_states_count += chunk_size
                    remaining_dp_counts -= chunk_size
                self.dump_log_file.write("current batch has {0} states, writing into a database with capacity {1}. after writing, there are {2} states in database\n".format(x.size(0), self.decoder_states_dataset.shape[0], self.decoder_states_count))

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            max_probs = None
            argmaxs = None
            for bd, tgt, is_single in batched:
                sample["target"] = tgt
                curr_prob = model.get_normalized_probs(
                    bd, log_probs=len(models) == 1, sample=sample
                ).data
                max_prob, argmax = torch.max(curr_prob, dim=-1)
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                    max_probs = max_prob
                    argmaxs = argmax
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                        max_probs = max_prob.new(orig_target.numel())
                        argmaxs = argmax.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(
                        curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt
                    )
                    probs[idx:end] = tgt_probs.view(-1)
                    max_probs[idx:end] = max_prob.view(-1)
                    argmaxs[idx:end] = argmax.view(-1)
                    idx = end
                sample["target"] = orig_target

            probs = probs.view(sample["target"].shape)
            states_shape = list(sample["target"].shape)
            states_shape.append(-1)
            states = decoder_out[0].detach().contiguous().view(states_shape)

            bsz = probs.size(0)
            if self.decoder_states_dump_dir is not None:
                for item_idx in range(bsz):
                    pad_filter_item = pad_filter[item_idx]
                    max_probs_item = max_probs[item_idx]
                    filtered_probs_item = max_probs_item[pad_filter_item].view(-1)
                    self.decoder_probs_dump_file.write(str(filtered_probs_item[0].item()))
                    for elem in filtered_probs_item[1:]:
                        self.decoder_probs_dump_file.write(" " + str(elem.item()))
                    self.decoder_probs_dump_file.write("\n")

            argmaxs = argmaxs.view(sample['target'].shape)
            ok_bad_tags = (argmaxs == sample['target'])
            if self.decoder_states_dump_file is not None:
                ok_bad_tags = ok_bad_tags.view(-1)
                ok_bad_tags = ok_bad_tags[pad_filter.view(-1)]
                for elem in ok_bad_tags:
                    tag = "OK" if elem.item() == True else "BAD"
                    self.ok_bad_tags_dump_file.write(tag + "\n")

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None:
                if torch.is_tensor(attn):
                    attn = attn.data
                else:
                    attn = attn[0]
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        hypos = []
        start_idxs = sample["start_indices"] if "start_indices" in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = (
                utils.strip_pad(sample["target"][i, start_idxs[i] :], self.pad)
                if sample["target"] is not None
                else None
            )
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i] : start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            states_i = states[i][start_idxs[i] : start_idxs[i] + tgt_len, :]
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample["net_input"]["src_tokens"][i],
                        sample["target"][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            if return_states:
                hypos.append(
                    [
                        {
                            "tokens": ref,
                            "score": score_i,
                            "attention": avg_attn_i,
                            "alignment": alignment,
                            "positional_scores": avg_probs_i,
                            "states": states_i,
                        }
                    ]
                )
            else:
                hypos.append(
                    [
                        {
                            "tokens": ref,
                            "score": score_i,
                            "attention": avg_attn_i,
                            "alignment": alignment,
                            "positional_scores": avg_probs_i,
                        }
                    ]
                )
        return hypos
