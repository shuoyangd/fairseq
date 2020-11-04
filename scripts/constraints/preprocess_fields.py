#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# @author Matt Post <post@cs.jhu.edu>

"""
Applies preprocessing to tab-delimited input fields, maintaining the tabs.
Since constrained decoding with fairseq-interactive takes input in the form
tab-delimited lines (first the input, then the first constraint, etc), this
allows source/target preprocessing to easily be applied separately.

Example usage (English -> German):

    echo -e "This is a test\tPrufung" \
    | preprocess_fields.py \
      [--source|-s SRC] \
      [--target|-t TRG] \
      [--tok] \
      [--norm] \
      [--bpe src-model [trg-model]] \
      [--sentencepiece src-model [trg-model]] \

where

- `--tok` and `--norm` trigger sacremoses tokenization and normalization,
  respectively, for language code {SRC} on field 1 and language code {TRG}
  on fields 2+
- `--source` and `--target` are ISO 639-1 language codes (only needed if
  applying tokenization and normalization)
- `-bpe` takes 1 or 2 models, applying model1 to field 1, and model2 to fields 2+.
  If no model2 is given, it is applied to all fields.
- `-sentencepiece` works in the same way.
"""

import sys
import sacremoses
import sentencepiece as sp


class Normalizer:
    def __init__(self, lang: str):
        self.lang = lang
        self.model = sacremoses.MosesPunctNormalizer(lang=self.lang)

    def __call__(self, segment):
        return self.model.normalize(segment)


class Tokenizer:
    def __init__(self, lang: str):
        self.lang = lang
        self.model = sacremoses.MosesTokenizer(lang=self.lang)

    def __call__(self, segment):
        return self.model.tokenize(segment, return_str=True)


class SPMModel:
    def __init__(self, model_file: str):
        self.model_file = model_file
        self.model = sp.SentencePieceProcessor(model_file=self.model_file)

    def __call__(self, segment):
        segmented = " ".join(self.model.encode(segment, out_type=str))
        return segmented


class BPEModel:
    def __init__(self, model_file: str):
        from subword_nmt import apply_bpe
        self.model_file = model_file
        self.model = apply_bpe.BPE(open(self.model_file))

    def __call__(self, segment):
        segmented = " ".join(self.model.segment(segment))
        return segmented

class FastBPEModel:
    def __init__(self, model_file: str):
        import fastBPE
        self.model_file = model_file
        self.model = fastBPE.fastBPE(model_file)

    def __call__(self, segment):
        segmented = self.model.apply([segment])[0]
        return segmented


def main(args):
    """Tokenizes, preserving tabs"""

    source_funcs = []
    target_funcs = []
    if args.normalize:
        source_funcs.append(Normalizer(args.source))
        target_funcs.append(Normalizer(args.target))

    if args.tokenize:
        source_funcs.append(Tokenizer(args.source))
        target_funcs.append(Tokenizer(args.target))

    if args.sentencepiece:
        source_funcs.append(SPMModel(args.sentencepiece[0]))
        if len(args.sentencepiece) > 1:
            target_funcs.append(SPMModel(args.sentencepiece[1]))
        else:
            target_funcs.append(source_funcs[-1])
    elif args.bpe:
        source_funcs.append(BPEModel(args.bpe[0]))
        if len(args.bpe) > 1:
            target_funcs.append(BPEModel(args.bpe[1]))
        else:
            target_funcs.append(source_funcs[-1])
    elif args.fastbpe:
        source_funcs.append(FastBPEModel(args.fastbpe[0]))
        if len(args.fastbpe) > 1:
            target_funcs.append(FastBPEModel(args.fastbpe[1]))
        else:
            target_funcs.append(source_funcs[-1])

    for line in sys.stdin:
        parts = line.rstrip().split("\t")
        for func in source_funcs:
            parts[0] = func(parts[0])
        for i, part in enumerate(parts[1:], 1):
            for func in target_funcs:
                parts[i] = func(parts[i])

        print(*parts, sep="\t", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s")
    parser.add_argument("--target", "-t")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--sentencepiece", nargs="+")
    parser.add_argument("--bpe", nargs="+")
    parser.add_argument("--fastbpe", nargs="+")
    args = parser.parse_args()

    main(args)
