# -*- coding: utf-8 -*-
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-01-05
#
# Distributed under terms of the MIT license.

from fairseq.data import FairseqDataset


class LabeledLanguagePairDataset(FairseqDataset):
  """
  Basically a LanguagePairDataset, but adds an extra label field to it so we can do regression.
  """

  def __init__(self, text_dataset, label_dataset):
    self.text_dataset = text_dataset
    self.label_dataset = label_dataset
    assert len(self.text_dataset) == len(self.label_dataset)

  def get_batch_shapes(self):
    return self.text_dataset.buckets

  def __getitem__(self, index):
    example = self.text_dataset[index]
    example["label"] = self.label_dataset[index]
    return example

  def __len__(self):
    assert len(self.text_dataset) == len(self.label_dataset)
    return len(self.text_dataset)

  def collater(self, samples, pad_to_length=None):
    batch = self.text_dataset.collater(samples, pad_to_length=pad_to_length)
    batch["label"] = self.label_dataset.collater(samples)
    return batch

  def num_tokens(self, index):
    return self.text_dataset.num_tokens(index)

  def size(self, index):
    return self.text_dataset.size(index)

  def ordered_indices(self):
    return self.text_dataset.ordered_indices()

  @property
  def supports_prefetch(self):
    return False  # raw label dataset doesn't support prefetch

  def filter_indices_by_size(self, indices, max_sizes):
    return self.text_dataset.filter_indices_by_size(indices, max_sizes)
