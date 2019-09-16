# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-09-11
#
# Distributed under terms of the MIT license.


import torch
import os


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

