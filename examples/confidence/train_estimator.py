# -*- coding: utf-8 -*-
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-01-04
#
# Distributed under terms of the MIT license.

import logging
import torch
from regressor import ConfidenceRegressor
from fairseq import checkpoint_utils, options, tasks, utils
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import pytorch_warmup as warmup

from labeled_language_pair_dataset import LabeledLanguagePairDataset
from sentence_level_label_dataset import SentenceLevelLabelDataset

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

LOG_INTERVAL = 1000

opt_parser = options.get_generation_parser()
group = opt_parser.add_argument_group("Confidence Model Train")
group.add_argument("--train-label-dataset", required=True, metavar="PATH", type=str,
                        help="A text file with log-scale confidence scores. One sentence per line. One score per token.")
group.add_argument("--valid-label-dataset", required=True, metavar="PATH", type=str,
                        help="A text file with log-scale confidence scores. One sentence per line. One score per token.")
group.add_argument("--model", required=True, metavar="PATH", type=str,
                        help="path to the translation model")
group.add_argument("--savedir", required=True, metavar="PATH", type=str,
                        help="path to the saved model")
group.add_argument("--hid-dim", type=int, default=512,
                        help="hidden dimension of the intermediate layers of regressor")
group.add_argument("--num-layers", type=int, default=5,
                        help="number of layers for regressor (must >=2)")
group.add_argument("--epochs", type=int, default=20)
group.add_argument("--gpu", action="store_true", default=False,
                        help="if gpu training should be used, set this option")
group.add_argument("--learning-rate", type=float, default=2e-5)
group.add_argument("--max-update-number", type=int, default=500000)

class ConfidenceRegressorWithModel(torch.nn.Module):

  def __init__(self, model, regressor):
    super(ConfidenceRegressorWithModel, self).__init__()
    self.model = model  # this should be a transformer model
    self.regressor = regressor

  @staticmethod
  def build_model(task, transformer_path, hid_dim, num_mid_layers):
    checkpoint = checkpoint_utils.load_checkpoint_to_cpu(transformer_path)
    model = task.build_model(checkpoint["args"])
    model.load_state_dict(checkpoint["model"], strict=True, args=checkpoint["args"])
    regressor = ConfidenceRegressor(checkpoint["args"].decoder_output_dim, hid_dim, 1, num_mid_layers=num_mid_layers)
    return ConfidenceRegressorWithModel(model, regressor)

  def forward(self, generator, sample):
    hypos = generator.generate(
      [self.model], sample, return_states=True, prefix_tokens=None, constraints=None,
    )  # this will be a list, each corresponding to a sentence
    states = [ hypo[0]["states"] for hypo in hypos ]
    states = torch.cat(states, dim=0)
    return self.regressor(states)


def load_label_dataset(path):
  data_file = open(path, 'r')
  labels = []
  for line in data_file:
    line = line.strip()
    scores = torch.Tensor([ float(token) for token in line.split(' ') ])
    labels.append(scores)
  data_file.close()
  return SentenceLevelLabelDataset(labels)

def train_iter(sample, estimator, generator, optimizer, mse_loss, pad_idx):
  optimizer.zero_grad()
  y = estimator(generator, sample)
  pad_filter = (sample["target"] != pad_idx).view(-1)
  label = torch.exp(sample["label"].view(-1)[pad_filter])
  loss = mse_loss(y.squeeze(1), label)
  loss.backward()
  optimizer.step()
  return loss

def valid_iter(sample, estimator, generator, mse_loss, pad_idx):
  with torch.no_grad():
    y = estimator(generator, sample)
    pad_filter = (sample["target"] != pad_idx).view(-1)
    label = torch.exp(sample["label"].view(-1)[pad_filter])
    loss = mse_loss(torch.exp(y.squeeze(1)), label)
  return loss

def validation(data_itr, estimator, generator, mse_loss, pad_idx, use_gpu=False):
  losses = []
  for sample in data_itr:
    sample = utils.move_to_cuda(sample) if use_gpu else sample
    losses.append(valid_iter(sample, estimator, generator, mse_loss, pad_idx).item())
  avg_loss = sum(losses) / len(losses)
  return avg_loss

def main(options):
  # prepare_dataset
  task = tasks.setup_task(options)
  task.load_dataset("train")
  task.load_dataset("valid")

  train_labels = load_label_dataset(options.train_label_dataset)
  valid_labels = load_label_dataset(options.valid_label_dataset)

  logging.info("loading labels... train")
  train_dataset = LabeledLanguagePairDataset(task.datasets["train"], train_labels)
  logging.info("loading labels... valid")
  valid_dataset = LabeledLanguagePairDataset(task.datasets["valid"], valid_labels)

  # build model
  logging.info("loading transformer checkpoint...")
  estimator = ConfidenceRegressorWithModel.build_model(task, options.model, options.hid_dim, options.num_layers-2)
  if options.gpu:
    estimator = estimator.cuda()

  # build optimizer
  optimizer = torch.optim.Adam(estimator.parameters(), lr=options.learning_rate)
  scheduler_steplr = CosineAnnealingLR(optimizer, options.max_update_number)
  scheduler_warmup = warmup.UntunedLinearWarmup(optimizer)
  optimizer.zero_grad()

  mse_loss = torch.nn.MSELoss()

  batch_itr = task.get_batch_iterator(
    dataset=train_dataset,
    max_tokens=options.max_tokens,
    max_sentences=options.batch_size,
    max_positions=utils.resolve_max_positions(
      task.max_positions(), estimator.model.max_positions()
    ),
    ignore_invalid_inputs=options.skip_invalid_size_inputs_valid_test,
    required_batch_size_multiple=options.required_batch_size_multiple,
    num_shards=options.num_shards,
    shard_id=options.shard_id,
    num_workers=1,
    # num_workers=options.num_workers,
    data_buffer_size=options.data_buffer_size,
  )
  valid_batch_itr = task.get_batch_iterator(
    dataset=valid_dataset,
    max_tokens=options.max_tokens,
    max_sentences=options.batch_size,
    max_positions=utils.resolve_max_positions(
      task.max_positions(), estimator.model.max_positions()
    ),
    ignore_invalid_inputs=options.skip_invalid_size_inputs_valid_test,
    required_batch_size_multiple=options.required_batch_size_multiple,
    num_shards=options.num_shards,
    shard_id=options.shard_id,
    num_workers=1,
    # num_workers=options.num_workers,
    data_buffer_size=options.data_buffer_size,
  )
  epoch_itr = batch_itr.next_epoch_itr(shuffle=True)
  valid_epoch_itr = valid_batch_itr.next_epoch_itr(shuffle=False)

  options.score_reference = True  # hard set to true, we assume a scorer is used from now on
  generator = task.build_generator(
    None, options,
  )  # first arg not going to be used anyway since we are building a scorer

  n_updates = 0
  for epoch_i in range(options.epochs):
    loss_accum = 0.0
    n_instances = 0
    for sample in epoch_itr:
      bsz = sample["target"].size(0)
      sample = utils.move_to_cuda(sample) if options.gpu else sample
      loss = train_iter(sample, estimator, generator, optimizer, mse_loss, task.tgt_dict.pad())
      loss_accum += loss
      n_instances += bsz
      n_updates += 1

      scheduler_steplr.step(scheduler_steplr.last_epoch+1)
      avg_warmup_factor = scheduler_warmup.dampen()
      lr = scheduler_steplr.get_last_lr()[0] * avg_warmup_factor
      if n_updates % LOG_INTERVAL == 0:
        logging.info("{0} instances visited, loss = {1}, lr = {2}".format(n_instances, loss_accum / LOG_INTERVAL, lr))
        loss_accum = 0.0

    logging.info("end of epoch {0}: {1} instances visited".format(epoch_i, n_instances))
    valid_loss = validation(valid_epoch_itr, estimator, generator, mse_loss, task.tgt_dict.pad(), options.gpu)
    logging.info("valid loss = {0}".format(valid_loss))
    torch.save(estimator, options.savedir + ".epoch{0}".format(epoch_i))
    epoch_itr = batch_itr.next_epoch_itr(shuffle=True)
    valid_epoch_itr = valid_batch_itr.next_epoch_itr(shuffle=False)


if __name__ == "__main__":
  args = options.parse_args_and_arch(opt_parser)
  main(args)