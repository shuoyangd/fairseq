# -*- coding: utf-8 -*-
#
# Copyright © 2020 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2020-09-07
#
# Distributed under terms of the MIT license.

import numpy as np
import pdb
import sys
import torch

from fairseq import data, options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_scorer import SequenceScorer
from fairseq.models.lstm import LSTMEncoder
from fairseq.modules import SalienceManager

N_LAYERS=12

def parallel_buffered_read(src_stream, tgt_stream, buffer_size):
    buffer = []
    for src_str, tgt_str in zip(src_stream, tgt_stream):
        buffer.append((src_str.strip(), tgt_str.strip()))
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer

def make_batches(src_lines, tgt_lines, args, task, max_positions):
    src_tokens = [
        task.source_dictionary.encode_line(src_str, add_if_not_exist=False).long()  # fairseq 0.9 change
        for src_str in src_lines
    ]
    tgt_tokens = [
        task.target_dictionary.encode_line(tgt_str, add_if_not_exist=False).long()  # fairseq 0.9 change
        for tgt_str in tgt_lines
    ]

    src_lengths = np.array([t.numel() for t in src_tokens])
    tgt_lengths = np.array([t.numel() for t in tgt_tokens])

    itr = task.get_batch_iterator(
        dataset=data.LanguagePairDataset(src_tokens, src_lengths, task.source_dictionary, \
                                         tgt_tokens, tgt_lengths, task.target_dictionary),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    return itr


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides))
    model = models[0]  # assume only one model for now
    model.eval()  # turn off dropout
    torch.no_grad()  # turn off backprop
    if use_cuda:
        model.cuda()
    pds = []
    attns = []

    # TODO: eliminate white baseline for the moment
    # used only when saliency == "integral"
    # background = torch.mean(model.encoder.embed_tokens.weights, dim=0)  # (emb_dim,)
    # if args.baseline == "b":
    #     background = background * 0.0
    # model.encoder.background = background

    def process_batch(batch):
        pds = []
        for layer_idx in range(N_LAYERS):
            if use_cuda:
                batch = utils.move_to_cuda(batch)
            net_input = {}
            new_bsz = 0
            slen = 0
            for key in batch['net_input'].keys():
                orig_size = batch['net_input'][key].size()
                orig_size = list(orig_size)
                if key == "src_tokens":
                    slen = orig_size[1]
                new_bsz = orig_size[0] * (slen+1)
                new_size = tuple([new_bsz] + orig_size[1:])  # repeat data by "the number of source word" times
                net_input[key] = batch['net_input'][key].expand(*new_size)  # TODO: this will not work for batch_size != 1
                net_input['pd_layer_idx'] = layer_idx

            target = batch['target']
            bsz, tlen = target.size()
            target = target.unsqueeze(2).expand(new_bsz, tlen, 1)

            decoder_out = model(**net_input)
            # sample argument is only used for adaptive softmax, so don't worry about it
            # we don't want *negative* log likelihood -- all the setting is to maximize objective
            probs = model.get_normalized_probs(decoder_out, log_probs=True, sample=None)  # (new_bsz, target_len, vocab)
            target_probs = torch.gather(probs, -1, target).view(bsz, -1, tlen)  # (bsz, source_len+1, target_len)
            pd = target_probs[:, 0, :] - target_probs[:, 1:, :]
            pd = pd.detach().cpu().transpose(1, 2)
            pds.append(pd)

        pd = torch.stack(pds, dim=1)  # (bsz, layers, tgt, src)
        if type(decoder_out[1]) == dict:
            attn = decoder_out[1]['attn']
        else:
            attn = decoder_out[1]
        attn = attn.detach().cpu()

        return pd, attn

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )
    num_batches = 0
    for inputs in parallel_buffered_read( \
            open(args.data + "/" + args.source_lang), \
            open(args.data + "/" + args.target_lang), \
            args.buffer_size
        ):
        src_inputs, tgt_inputs = zip(*inputs)
        for batch in make_batches(src_inputs, tgt_inputs, args, task, max_positions):
            pd, attn = process_batch(batch)
            # loop over each layer
            for i in range(pd.size(1)):
                pds.append(pd[:, i])
            attns.append(attn)
            num_batches += 1
            if num_batches % 10 == 0:
                sys.stderr.write(".")
                sys.stderr.flush()

    torch.save(pds, open(args.out + ".pd", 'wb'))
    torch.save(attns, open(args.out + ".at", 'wb'))


if __name__ == '__main__':
    parser = options.get_generation_parser(True)
    parser.add_argument("--out", metavar="PATH", help="")
    args = options.parse_args_and_arch(parser)
    main(args)
