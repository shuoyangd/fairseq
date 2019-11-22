# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-09-11
#
# Distributed under terms of the MIT license.


import torch
import os
import pdb


def read_subjs_data(filename):
  subjs_file = open(filename)
  subjs = []
  for line in subjs_file:
    subjs.append(torch.LongTensor([int(idx) for idx in line.strip().split()]))
  return subjs


def read_tags(tag_file_path):
  tags = []
  with open(tag_file_path) as f:
    for line in f:
      tags.append([int(line.strip())])
  return torch.Tensor(tags)


def read_subjs_data_set(filename):
  subjs_file = open(filename)
  subjs = []
  attrs = []
  for line in subjs_file:
    fields = line.strip().split(" ||| ")
    subjs_in_line = torch.IntTensor(eval(fields[0]))
    attrs_in_line = torch.IntTensor(eval(fields[1]))
    if attrs_in_line.size(0) == 0:
      pdb.set_trace()
    subjs.append(subjs_in_line)
    attrs.append(attrs_in_line)

  return subjs, attrs


class SentCorpus(object):
  def __init__(self, path, dictionary, append_eos=True):
    self.dictionary = dictionary
    self.test = self.tokenize(path, append_eos)

  def tokenize(self, path, append_eos=True):
    assert os.path.exists(path)
    ids = []
    with open(path, 'r') as f:
      for line in f:
        sent = self.dictionary.encode_line(line, add_if_not_exist=False, append_eos=append_eos)
        ids.append(sent)

    return ids

