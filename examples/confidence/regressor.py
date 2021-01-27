# -*- coding: utf-8 -*-
#
# Copyright © 2020 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2020-07-05
#
# Distributed under terms of the MIT license.

import argparse
import h5py
import logging
import pdb
import torch
import time
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="")
opt_parser.add_argument("--mode", choices=["train", "inference"], required=True, help="")
opt_parser.add_argument("--model", type=str, metavar="PATH", help="")
opt_parser.add_argument("--in-dim", type=int, default=512, help="")
opt_parser.add_argument("--hid-dim", type=int, default=512, help="")
opt_parser.add_argument("--state-dir", type=str, metavar="PATH", required=True, help="")
opt_parser.add_argument("--label-dir", type=str, metavar="PATH", required=True, help="")
opt_parser.add_argument("--batch-size", type=int, default=64, help="")
opt_parser.add_argument("--epochs", type=int, default=5, help="")
opt_parser.add_argument("--gpu", action="store_true", default=False, help="")


LOG_INTERVAL = 1

def Linear(in_features, out_features, bias=True, uniform=True, dropout=0.0):
  m = nn.Linear(in_features, out_features, bias)
  if uniform:
    nn.init.xavier_uniform_(m.weight)
  else:
    nn.init.xavier_normal_(m.weight)
  if bias:
    nn.init.constant_(m.bias, 0.)
  return m


def Linear2(in_features, out_features, dropout=0.0):
  w = nn.Parameter(torch.Tensor(out_features, in_features))
  nn.init.normal_(w, mean=0, std=in_features ** -0.5)
  m = nn.Linear(in_features, out_features, bias=False)
  m.weight = w
  return m


class ConfidenceRegressor(torch.nn.Module):

  def __init__(self, D_in, H, D_out, num_mid_layers=3):
    super(ConfidenceRegressor, self).__init__()
    # self.linear1 = Linear(D_in, H, bias=False, uniform=False)
    # self.linear2 = Linear2(H, D_out)
    self.linear1 = nn.Linear(D_in, H)
    self.layernorm1 = nn.LayerNorm(H)
    self.middle = []
    self.middle_layernorm = []
    for i in range(num_mid_layers):
        self.middle.append(nn.Linear(H, H))
        self.middle_layernorm.append(nn.LayerNorm(H))
    self.middle = nn.ModuleList(self.middle)
    self.middle_layernorm = nn.ModuleList(self.middle_layernorm)
    self.linear2 = nn.Linear(H, D_out)

  def forward(self, x):
    h = self.linear1(x)
    h = self.layernorm1(h)
    h = torch.tanh(h)
    for layer, layernorm in zip(self.middle, self.middle_layernorm):
        h = layer(h)
        h = layernorm(h)
        h = torch.tanh(h)
    y_pred = torch.sigmoid(self.linear2(h))
    return y_pred


def train_iter(batch, obj, estimator, optimizer):
    end1 = time.time()
    batch = torch.Tensor(batch)
    obj = torch.Tensor(obj)
    if options.gpu:
      batch = batch.cuda()
      obj = obj.cuda()
    end2 = time.time()
    logging.debug("moving batch cost {0} seconds".format(end2 - end1))
    y = estimator(batch)
    end3 = time.time()
    logging.debug("forward pass cost {0} seconds".format(end3 - end2))
    if len(y) != len(obj):
      logging.warning("input has length {0} while there is only {1} objects".format(len(y), len(obj)))
      truncated_len = min(len(y), len(obj))
      y = y[:truncated_len]
      obj = obj[:truncated_len]
    # loss = -torch.sum(y[torch.arange(options.batch_size).long(), obj]) / len(y)
    mse = nn.MSELoss()
    loss = mse(torch.exp(y.squeeze(1)), torch.exp(obj))

    loss.backward()
    end4 = time.time()
    logging.debug("backward pass cost {0} seconds".format(end4 - end3))
    optimizer.step()
    end5 = time.time()
    logging.debug("param update cost {0} seconds".format(end5 - end4))

    return loss

def train(options):
  estimator = ConfidenceRegressor(options.in_dim, options.hid_dim, 1)
  if options.gpu:
    estimator = estimator.cuda()

  optimizer = optim.Adam(estimator.parameters(), lr=5e-4)
  # if options.gpu:
  #   optimizer = optimizer.cuda()

  scheduler_steplr = CosineAnnealingLR(optimizer, 20000)
  scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1, after_scheduler=scheduler_steplr)
  optimizer.zero_grad()
  optimizer.step()

  # pre-load labels
  label_file = open(options.label_dir, 'r')
  labels = label_file.readlines()

  # build data
  batch = []
  obj = []
  n_updates = 0
  for epoch_i in range(options.epochs):
    state_file = h5py.File(options.state_dir)

    loss_accum = 0.0
    n_instances = 0
    # for state, label in zip(state_file["decoder_states"], label_file):
    for idx in torch.randperm(len(labels)):
      state = state_file["decoder_states"][idx]
      label = labels[idx]
      scheduler_warmup.step(n_updates+1)
      label = label.strip()
      start = time.time()
      if len(batch) != options.batch_size:
        batch.append(state)
        obj.append(float(label))
      else:
        end1 = time.time()
        logging.debug("building a batch used {0} seconds".format(end1 - start))
        loss = train_iter(batch, obj, estimator, optimizer)
        loss_accum += loss
        n_instances += options.batch_size
        n_updates += 1
        lr = scheduler_warmup.get_lr()[0]
        if n_updates % LOG_INTERVAL == 0:
          logging.info("{0} instances visited, loss = {1}, lr = {2}".format(n_instances, loss_accum / LOG_INTERVAL, lr))
          loss_accum = 0.0

        batch = []
        obj = []

    if len(batch) != 0 and len(obj) != 0:
      loss_accum += train_iter(batch, obj, estimator, optimizer)
      n_instances += len(batch)
      n_updates += 1

    logging.info("end of epoch {0}: {1} instances visited".format(epoch_i, n_instances))
    # TODO: dev
    torch.save(estimator, options.model)

def inference(options):
  estimator = torch.load(options.model)
  state_file = h5py.File(options.state_dir)
  label_file = open(options.label_dir, 'w')

  batch = []
  for state in zip(state_file["decoder_states"]):
    if len(batch) != options.batch_size:
      batch.append(state)
    else:
      batch = torch.Tensor(batch)
      if options.gpu:
        batch = batch.cuda()
      y_pred = estimator(batch)
      for conf in y_pred[:, 0, 0]:
        label_file.write(str(conf.item()) + "\n")
      batch = []

def main(options):
  if options.mode == "train":
    train(options)
  elif options.mode == "inference":
    inference(options)
  else:
    raise NotImplementedError

if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
