#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import fileinput
import logging
import math
import os
import re
import sys
import time
from collections import namedtuple

import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    if args.prefix_size > 0 or args.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        prefixes = None
        if args.prefix_size > 0:
            prefixes = torch.zeros((len(lines), args.prefix_size)).long()

        sys_lines = []
        ref_lines = []
        for i, line in enumerate(lines):
            fields = line.split("\t")
            assert (
                (args.prefix_size == 0 and args.n_ensemble_views == 1) or len(fields) >= 2
            ), "--prefix-size and --no-ensemble-views both require a second tab-delimited field " + \
               "(ensemble views comes before target prefixes)"

            # consume the multiview ensemble fields first,
            # which comes before target prefixes
            if args.n_ensemble_views > 1:
                sys_lines.append(fields.pop(1))
                if args.n_ensemble_views == 3:
                    ref_lines.append(fields.pop(1))

            # then it's target prefix
            if args.prefix_size > 0:
                prefix = fields.pop(1)
                assert (
                    len(prefix.split()) >= args.prefix_size
                ), "prefix must have at at least --prefix-size tokens"
                prefixes[i] = task.target_dictionary.encode_line(
                    encode_fn(prefix),
                    append_eos=False,
                    add_if_not_exist=False,
                )[:args.prefix_size]

            lines[i] = fields.pop(0)

            # If any fields remain, use them as constraints
            if len(fields) > 0:
                batch_constraints[i] = [
                    task.target_dictionary.encode_line(
                        encode_fn(constraint),
                        append_eos=False,
                        add_if_not_exist=False,
                    )
                    for constraint in fields
                ]

    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]

    if args.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)

    if args.n_ensemble_views > 1:
        sys_tokens = [
            task.source_dictionary.encode_line(
                encode_fn(src_str), add_if_not_exist=False
            ).long()
            for src_str in sys_lines
        ]
        sys_lengths = [ t.numel() for t in sys_tokens ]
        if args.n_ensemble_views == 3:
            ref_tokens = [
                task.source_dictionary.encode_line(
                    encode_fn(src_str), add_if_not_exist=False
                ).long()
                for src_str in ref_lines
            ]
            ref_lengths = [ t.numel() for t in ref_tokens ]

        sys_itr = task.get_batch_iterator(
            dataset=task.build_dataset_for_inference(
                sys_tokens, sys_lengths, constraints=constraints_tensor
            ),
            max_tokens=args.max_tokens,
            max_sentences=args.batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        ).next_epoch_itr(shuffle=False)
        if args.n_ensemble_views == 3:
            ref_itr = task.get_batch_iterator(
                dataset=task.build_dataset_for_inference(
                    ref_tokens, ref_lengths, constraints=constraints_tensor
                ),
                max_tokens=args.max_tokens,
                max_sentences=args.batch_size,
                max_positions=max_positions,
                ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            ).next_epoch_itr(shuffle=False)

    if args.n_ensemble_views == 2:
        for src_batch, sys_batch in zip(itr, sys_itr):
            ids = src_batch["id"]
            src_tokens = src_batch["net_input"]["src_tokens"]
            sys_tokens = sys_batch["net_input"]["src_tokens"]
            src_lengths = src_batch["net_input"]["src_lengths"]
            sys_lengths = sys_batch["net_input"]["src_lengths"]
            constraints = src_batch.get("constraints", None)

            Batch = namedtuple("Batch", "ids src_tokens sys_tokens src_lengths sys_lengths constraints prefixes")
            yield Batch(
                ids=ids,
                src_tokens=src_tokens,
                sys_tokens=sys_tokens,
                src_lengths=src_lengths,
                sys_lengths=sys_lengths,
                constraints=constraints,
                prefixes=prefixes,
            )

    elif args.n_ensemble_views == 3:
        for src_batch, sys_batch, ref_batch in zip(itr, sys_itr, ref_itr):
            ids = src_batch["id"]
            src_tokens = src_batch["net_input"]["src_tokens"]
            sys_tokens = sys_batch["net_input"]["src_tokens"]
            ref_tokens = ref_batch["net_input"]["src_tokens"]
            src_lengths = src_batch["net_input"]["src_lengths"]
            sys_lengths = sys_batch["net_input"]["src_lengths"]
            ref_lengths = ref_batch["net_input"]["src_lengths"]
            constraints = src_batch.get("constraints", None)

            Batch = namedtuple("Batch", "ids src_tokens sys_tokens ref_tokens src_lengths sys_lengths ref_lengths constraints prefixes")
            yield Batch(
                ids=ids,
                src_tokens=src_tokens,
                sys_tokens=sys_tokens,
                ref_tokens=ref_tokens,
                src_lengths=src_lengths,
                sys_lengths=sys_lengths,
                ref_lengths=ref_lengths,
                constraints=constraints,
                prefixes=prefixes,
            )

    else:
        for batch in itr:
            ids = batch["id"]
            src_tokens = batch["net_input"]["src_tokens"]
            src_lengths = batch["net_input"]["src_lengths"]
            constraints = batch.get("constraints", None)

            Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints prefixes")
            yield Batch(
                ids=ids,
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                constraints=constraints,
                prefixes=prefixes,
            )


def main(args):
    start_time = time.time()
    total_translate_time = 0

    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.batch_size is None:
        args.batch_size = 1

    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not args.batch_size or args.batch_size <= args.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

    # Initialize generator
    generator = task.build_generator(models, args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    if args.constraints:
        logger.info("Enabled decoding with lexical beam constraints.")
        if tokenizer is not None or bpe is not None:
            logger.warning(
                "NOTE: Applying source-side preprocessing to the constraints. "
                "If you used different target side preprocessing, you need to "
                "apply it outside of fairseq."
            )

    if args.buffer_size > 1:
        logger.info("Sentence buffer size: %s", args.buffer_size)
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Type the input sentence and press return:")
    start_id = 0
    for inputs in buffered_read(args.input, args.buffer_size):
        results = []
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            sys_tokens = batch.sys_tokens if hasattr(batch, "sys_tokens") else None
            ref_tokens = batch.ref_tokens if hasattr(batch, "ref_tokens") else None
            src_lengths = batch.src_lengths
            sys_lengths = batch.sys_lengths if hasattr(batch, "sys_lengths") else None
            ref_lengths = batch.ref_lengths if hasattr(batch, "ref_lengths") else None
            constraints = batch.constraints
            prefixes = batch.prefixes
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()
                if prefixes is not None:
                    prefixes = prefixes.cuda()
                sys_tokens = sys_tokens.cuda() if sys_tokens is not None else None
                ref_tokens = ref_tokens.cuda() if ref_tokens is not None else None
                sys_lengths = sys_lengths.cuda() if sys_lengths is not None else None
                ref_lengths = ref_lengths.cuda() if ref_lengths is not None else None

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }
            if sys_tokens is not None and sys_lengths is not None:
                sample["net_input"]["sys_tokens"] = sys_tokens
                sample["net_input"]["sys_lengths"] = sys_lengths
            if ref_tokens is not None and ref_lengths is not None:
                sample["net_input"]["ref_tokens"] = ref_tokens
                sample["net_input"]["ref_lengths"] = ref_lengths
            translate_start_time = time.time()
            translations = task.inference_step(
                generator, models, sample, constraints=constraints, prefix_tokens=prefixes
            )
            translate_time = time.time() - translate_start_time
            total_translate_time += translate_time
            list_constraints = [[] for _ in range(bsz)]
            if args.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                constraints = list_constraints[i]
                results.append(
                    (
                        start_id + id,
                        src_tokens_i,
                        hypos,
                        {
                            "constraints": constraints,
                            "time": translate_time / len(translations),
                        },
                    )
                )

        # sort output to match input order
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                print("S-{}\t{}".format(id_, src_str))
                print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))
                for constraint in info["constraints"]:
                    print(
                        "C-{}\t{}".format(
                            id_, tgt_dict.string(constraint, None)
                        )
                    )

            # Process top predictions
            for hypo in hypos[: min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                tok_hypo_str = tgt_dict.string(
                    hypo["tokens"].int().cpu(), None, get_symbols_to_strip_from_output(generator)
                )

                score = hypo["score"] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                print("H-{}\t{}\t{}".format(id_, score, tok_hypo_str))
                # detokenized hypothesis
                print("D-{}\t{}\t{}".format(id_, score, hypo_str))
                print(
                    "P-{}\t{}".format(
                        id_,
                        " ".join(
                            map(
                                lambda x: "{:.4f}".format(x),
                                # convert from base e to base 2
                                hypo["positional_scores"].div_(math.log(2)).tolist(),
                            )
                        ),
                    )
                )
                if args.print_alignment:
                    alignment_str = " ".join(
                        ["{}-{}".format(src, tgt) for src, tgt in alignment]
                    )
                    print("A-{}\t{}".format(id_, alignment_str))

        # update running id_ counter
        start_id += len(inputs)

    logger.info(
        "Total time: {:.3f} seconds; translation time: {:.3f}".format(
            time.time() - start_time, total_translate_time
        )
    )


def cli_main():
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
