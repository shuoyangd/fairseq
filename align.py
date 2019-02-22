#!/usr/bin/env python3 -u
"""
align source and target sentence with a model
"""

import numpy as np
import pdb
import torch

from fairseq import data, options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_scorer import SequenceScorer


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
        tokenizer.Tokenizer.tokenize(tgt_str, task.source_dictionary, add_if_not_exist=False).long()
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
    saliencies = []
    attns = []

    def compute_saliency(grad):
        saliency = grad.norm(dim=-1)
        saliencies.append(saliency / torch.sum(saliency, dim=1))

    def guided_hook(module, grad_in, grad_out):
        return tuple([ torch.clamp(grad, min=0.0) for grad in grad_in ])

    def process_batch(batch):
        net_input = batch['net_input']
        src_tokens = net_input['src_tokens']
        x = model.encoder.embed_scale * model.encoder.embed_tokens(src_tokens)
        if model.encoder.embed_positions is not None:  # TODO: test w/ or w/o positional embedding?
            x = x + model.encoder.embed_positions(src_tokens)
        x.register_hook(compute_saliency)
        net_input['src_emb'] = x

        if args.saliency == "guided":
            for module in model.modules():
                module.register_backward_hook(guided_hook)

        decoder_out = model(**net_input)
        probs = model.get_normalized_probs(decoder_out, log_probs=False, sample=batch)  # (batch_size, target_len, vocab)
        grad_outputs = torch.zeros_like(probs)

        target = batch['target']
        bsz, tlen = target.size()
        target = target.view(bsz, tlen, 1)
        batch_idx = torch.arange(bsz).view(bsz, -1).expand(bsz, tlen)
        tlen_idx = torch.arange(tlen).view(-1, tlen).expand(bsz, tlen)
        grad_outputs[batch_idx, tlen_idx, target] = 1
        probs.backward(gradient=grad_outputs)
        model.zero_grad()

        attn = decoder_out[1]['attn']
        return attn

        """
        target = batch['target']
        bsz, tlen = target.size()
        target = target.view(bsz, tlen, 1)
        target_probs = torch.gather(probs, -1, target).view(bsz, tlen)  # (batch_size * target_len)
        for i in range(bsz):
            for j in range(tlen):
                target_probs[i, j].backward(retain_graph=True)
                pdb.set_trace()
                model.zero_grad()
        """

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )
    for inputs in parallel_buffered_read( \
            open(args.data[0] + "/" + args.source_lang), \
            open(args.data[0] + "/" + args.target_lang), \
            args.buffer_size
        ):
        src_inputs, tgt_inputs = zip(*inputs)
        for batch in make_batches(src_inputs, tgt_inputs, args, task, max_positions):
            pdb.set_trace()
            attns.append(process_batch(batch))

if __name__ == '__main__':
    parser = options.get_generation_parser(True)
    parser.add_argument("--saliency", choices=["plain", "guided", "deconv"], help="")
    args = options.parse_args_and_arch(parser)
    main(args)
