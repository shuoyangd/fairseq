# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import os, re
import pdb

import torch
from multiprocessing import Pool

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()

def tokenize_line_char(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    char_list = [ merge_reserve(list(s)) for s in line.split() ]
    return char_list

def tokenize_line_bpe(line, bpe_module):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    bpe_list = [ bpe_module.segment_tokens([w]) for w in line.split() ]
    return bpe_list

def merge_reserve(char_list):
  ret = []
  reserve_buf = ""
  reserving_and = False
  reserving_at = False
  for idx, char in enumerate(char_list):
    if char == '&' and not reserving_and:
      reserve_buf += char
      reserving_and = True
    elif char == '@' and not reserving_at:
      reserve_buf += char
      reserving_at = True
    elif char == ';' and reserving_and:
      reserve_buf += char
      reserving_and = False
      ret.append(reserve_buf)
      reserve_buf = ""
    elif char == '@' and reserving_at:
      reserve_buf += char
      reserving_at = False
      ret.append(reserve_buf)
      reserve_buf = ""
    elif reserve_buf:
      reserve_buf += char
    else:
      ret.append(char)
  return ret

def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos) # search where this character begins

class Tokenizer:

    @staticmethod
    def add_file_to_dictionary_single_worker(filename, tokenize, eos_word, worker_id=0, num_workers=1):
        counter = Counter()
        with open(filename, 'r') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f) # drop first incomplete line
            line = f.readline()
            while line:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(counter):
            for w, c in counter.items():
                dict.add_symbol(w, c)
        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(pool.apply_async(
                    Tokenizer.add_file_to_dictionary_single_worker,
                    (filename, tokenize, dict.eos_word, worker_id, num_workers)
                ))
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(Tokenizer.add_file_to_dictionary_single_worker(filename, tokenize, dict.eos_word))

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line,
                            append_eos=True, reverse_order=False,
                            offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()
        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])
        with open(filename, 'r') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = Tokenizer.tokenize(
                    line=line,
                    dict=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, 'r') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets

    @staticmethod
    def tokenize(line, dict, tokenize=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False):
        words = tokenize(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dict.add_symbol(word)
            else:
                idx = dict.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dict.eos_index
        return ids


class CharTokenizer:

    @staticmethod
    def add_file_to_dictionary_single_worker(filename, tokenize, eos_word, worker_id=0, num_workers=1):
        counter = Counter()
        with open(filename, 'r') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f) # drop first incomplete line
            line = f.readline()
            while line:
                for word in tokenize(line):
                    for char in word:
                        counter.update([char])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(counter):
            for w, c in counter.items():
                dict.add_symbol(w, c)
        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(pool.apply_async(
                    CharTokenizer.add_file_to_dictionary_single_worker,
                    (filename, tokenize, dict.eos_word, worker_id, num_workers)
                ))
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(CharTokenizer.add_file_to_dictionary_single_worker(filename, tokenize, dict.eos_word))

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line_char,
                            append_eos=True, reverse_order=False,
                            offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()
        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])
        with open(filename, 'r') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = CharTokenizer.tokenize(
                    line=line,
                    dict=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def tokenize(line, dict, tokenize=tokenize_line_char, add_if_not_exist=True,
                 consumer=None, append_eos=True, append_eow=True, reverse_order=False):
        chars = tokenize(line)
        if reverse_order:
            chars = list(reversed(chars))
        nwords = len(chars)
        max_word_len = max(list(map(lambda x: len(x), chars)) + [0])
        if max_word_len == 0:  # empty sentence, should not proceed further
            return torch.IntTensor([])

        ids = torch.IntTensor(\
                nwords + 1 if append_eos else nwords, \
                max_word_len + 1 if append_eow else max_word_len \
              )
        ids.fill_(dict.pad_index)

        for i, word in enumerate(chars):
            nchars = len(word)
            for j, char in enumerate(word):
                if add_if_not_exist:
                    idx = dict.add_symbol(char)
                else:
                    idx = dict.index(char)
                if consumer is not None:
                    consumer(char, idx)
                ids[i, j] = idx
            if append_eow:
                ids[i, nchars] = dict.eow_index
        if append_eos:
            ids[nwords, 0] = dict.eos_index
        return ids

