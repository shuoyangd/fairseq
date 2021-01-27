# -*- coding: utf-8 -*-
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-01-06
#
# Distributed under terms of the MIT license.

import math
import pdb
import torch

from fairseq.data import FairseqDataset

class SentenceLevelLabelDataset(FairseqDataset):
  def __init__(self, labels):
    super().__init__()
    self.labels = labels

  def __getitem__(self, index):
    return self.labels[index]

  def __len__(self):
    return len(self.labels)

  def collater(self, samples):
    max_len = max([ len(sample["label"]) for sample in samples ])
    new_samples = []
    for sample in samples:
      scores = sample["label"].tolist()
      new_scores = scores + ([-math.inf] * (max_len - len(scores)))
      new_samples.append(torch.Tensor(new_scores))
    return torch.stack(new_samples, dim=0)