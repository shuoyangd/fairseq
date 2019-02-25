# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-02-19
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
from fairseq.models import SaliencyManager
from fairseq.models.lstm import LSTMEncoder

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
        tokenizer.Tokenizer.tokenize(src_str, task.source_dictionary, add_if_not_exist=False).long()
        for src_str in src_lines
    ]
    tgt_tokens = [
        tokenizer.Tokenizer.tokenize(tgt_str, task.target_dictionary, add_if_not_exist=False).long()
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
    if type(model.encoder) == LSTMEncoder:
        model.encoder.training = False
        model.decoder.training = False
    else:
        model.eval()  # turn off dropout, will kill cudnn rnn backward
    if use_cuda:
        model.cuda()
    saliencies = []
    attns = []

    def guided_hook(module, grad_in, grad_out):
        return tuple([ torch.clamp(grad, min=0.0) for grad in grad_in ])

    def process_batch(batch):
        if use_cuda:
            batch = utils.move_to_cuda(batch)
        net_input = batch['net_input']
        net_input['smoothing_factor'] = args.smoothing_factor
        if args.abs:
            net_input['abs_saliency'] = True

        """
        src_tokens = net_input['src_tokens']
        x = model.encoder.embed_scale * model.encoder.embed_tokens(src_tokens)
        if model.encoder.embed_positions is not None:  # TODO: test w/ or w/o positional embedding?
            x = x + model.encoder.embed_positions(src_tokens)
        # x.register_hook(compute_saliency)
        net_input['src_emb'] = x
        """

        if args.saliency == "guided":
            for module in model.modules():
                if type(module) == torch.nn.modules.linear.Linear:
                    module.register_backward_hook(guided_hook)

        target = batch['target']
        bsz, tlen = target.size()
        target = target.view(bsz, tlen, 1)

        for sample_i in range(args.n_samples):
            decoder_out = model(**net_input)
            probs = model.get_normalized_probs(decoder_out, log_probs=False, sample=batch)  # (batch_size, target_len, vocab)
            target_probs = torch.gather(probs, -1, target).view(bsz, tlen)  # (batch_size * target_len)
            for i in range(bsz):
                for j in range(tlen):
                    target_probs[i, j].backward(retain_graph=True)
                    model.zero_grad()

        # single sentence saliency will be a list with (tgt * n_samples) of (bsz, src)
        saliency = torch.stack(SaliencyManager.single_sentence_saliency, dim=1)  # (bsz, tgt * n_samples, src)
        bsz = saliency.size(0)
        saliency = saliency.view(bsz, args.n_samples, tlen, -1)  # (bsz, n_samples, tgt, src)
        saliency = torch.mean(saliency, dim=1)  # (bsz, tgt, src)
        if type(decoder_out[1]) == dict:
            attn = decoder_out[1]['attn']
        else:
            attn = decoder_out[1]
        SaliencyManager.clear_saliency()
        return saliency, attn

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )
    num_batches = 0
    for inputs in parallel_buffered_read( \
            open(args.data[0] + "/" + args.source_lang), \
            open(args.data[0] + "/" + args.target_lang), \
            args.buffer_size
        ):
        src_inputs, tgt_inputs = zip(*inputs)
        for batch in make_batches(src_inputs, tgt_inputs, args, task, max_positions):
            pdb.set_trace()
            saliency, attn = process_batch(batch)
            saliencies.append(saliency)
            attns.append(attn)
            num_batches += 1
            if num_batches % 10 == 0:
                sys.stderr.write(".")
                sys.stderr.flush()

    torch.save(saliencies, open(args.out + ".sa", 'wb'))
    torch.save(attns, open(args.out + ".at", 'wb'))


if __name__ == '__main__':
    parser = options.get_generation_parser(True)
    parser.add_argument("--saliency", choices=["plain", "guided", "deconv"], help="")
    parser.add_argument("--out", metavar="PATH", help="")
    parser.add_argument("--smoothing-factor", "-sf", type=float, default=0.0, help="")
    parser.add_argument("--n-samples", "-sn", type=int, default=1, help="")
    parser.add_argument("--abs", action='store_true', default=False, help="")
    args = options.parse_args_and_arch(parser)
    main(args)
