# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-07-03
#
# Distributed under terms of the MIT license.

import argparse
import data
import logging
import os.path
import pdb
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
# cudnn backward cannot be called at eval mode
import torch.backends.cudnn as cudnn
cudnn.enabled = False

from fairseq import options, progress_bar, tasks, utils
from fairseq.utils import import_user_module
from fairseq.models.transformer import Linear

from utils import batchify2
from salience import SalienceManager
from salience import SalienceType

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

"""
opt_parser = argparse.ArgumentParser(description="fine-tune a filter for salience test on awd language model")
opt_parser.add_argument("--model", type=str, metavar="PATH", required=True, help="path to the saved model")
opt_parser.add_argument("--outdir", type=str, metavar="PATH", required=True, help="path to the finetuned model")
opt_parser.add_argument("--data-prefix", type=str, metavar="PATH", required=True, help="path to training and dev data (without the .prefx.txt, tag.txt suffix)")
opt_parser.add_argument("--dict-data", type=str, metavar="PATH", required=True, help="path to the data used to build dict")
opt_parser.add_argument("--batch-size", type=int, default=10, help="test batch size")
opt_parser.add_argument("--cuda", action='store_true', default=False, help="use cuda")

opt_parser.add_argument("--max-epoch", type=int, default=10)
opt_parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam"], default="SGD")
opt_parser.add_argument("--momentum", type=float, default=0.99, help="momentum for SGD")
opt_parser.add_argument("--learning-rate", '-lr', type=float, default=0.1)
"""


def model_load(path, task, model_overrides=False):
  models, args = utils.load_ensemble_for_inference(
    path.split(':'), task, model_arg_overrides=model_overrides,
  )
  return models, args


"""
def get_normalized_probs(output, weight, bias=None, log_probs=True):
  logits = torch.matmul(output, weight.transpose(0, 1))  # (samples, vocab_size)
  if bias is not None:
    logits += bias
  if log_probs:
    return F.log_softmax(logits, dim=-1)
  else:
    return F.softmax(logits, dim=-1)
"""


def finetune(prefix_data, tag_data, model, outdir, optimizer, epoch_n, cuda=False):
  iter_ = 0
  for (prefix_batch, prefix_mask), (tag_batch, _) in \
      zip(zip(prefix_data[0], prefix_data[1]), zip(tag_data[0], tag_data[1])):

    optimizer.zero_grad()
    batch_size = tag_batch.size(1)
    padded_seq_len = prefix_batch.size(0)

    if cuda:
        prefix_batch = prefix_batch.cuda()
        prefix_mask = prefix_mask.cuda()
        tag_batch = tag_batch.cuda()

    output, _ = model(prefix_batch.transpose(0, 1), None)
    probs = model.get_normalized_probs([output], log_probs=True).transpose(0, 1)  # (padded_seq_len, batch_size, vocab_size)

    final_prefix_index = (torch.sum(prefix_mask, dim=0).unsqueeze(0) - 1).clamp_(0)  # (1, bsz) TODO: there is -1 index
    final_prefix_index = final_prefix_index.unsqueeze(2).expand(-1, -1, probs.size(-1))  # (1, batch_size, vocab_size)
    probs = torch.gather(probs, 0, final_prefix_index).squeeze(0)  # (bsz, vocab_size)

    loss = -torch.sum(torch.gather(probs, 1, tag_batch.transpose(0, 1)))  # (bsz, 1)
    # loss = -torch.sum(probs[:, tag_batch])
    loss.backward()
    optimizer.step()

    iter_ += 1
    if iter_ % 100 == 0:
      print("training loss at {0} is {1}".format(iter_, loss.item() / batch_size))

  torch.save(model, "{0}.epoch{1}".format(outdir, epoch_n))
  return model


def validate(prefix_data, tag_data, model, cuda=False):
  raw_loss = 0.0
  n_samples = 0
  for (prefix_batch, prefix_mask), (tag_batch, _) in \
      zip(zip(prefix_data[0], prefix_data[1]), zip(tag_data[0], tag_data[1])):

    batch_size = tag_batch.size(1)
    padded_seq_len = prefix_batch.size(0)
    n_samples += batch_size

    if cuda:
        prefix_batch = prefix_batch.cuda()
        prefix_mask = prefix_mask.cuda()
        tag_batch = tag_batch.cuda()

    output, _ = model(prefix_batch.transpose(0, 1), None)
    probs = model.get_normalized_probs([output], log_probs=True).transpose(0, 1)  # (padded_seq_len, batch_size, vocab_size)

    final_prefix_index = (torch.sum(prefix_mask, dim=0).unsqueeze(0) - 1).clamp_(0)  # (1, bsz) TODO: there is -1 index
    final_prefix_index = final_prefix_index.unsqueeze(2).expand(-1, -1, probs.size(-1))  # (1, batch_size, vocab_size)
    probs = torch.gather(probs, 0, final_prefix_index).squeeze(0)  # (bsz, vocab_size)

    loss = -torch.sum(torch.gather(probs, 1, tag_batch.transpose(0, 1)))  # (bsz, 1)
    raw_loss += loss

  return raw_loss / n_samples


def main(parsed_args):
  torch.cuda.set_device(0)  # TODO: temporary

  import_user_module(parsed_args)
  print(parsed_args)
  use_cuda = torch.cuda.is_available() and not parsed_args.cpu
  task = tasks.setup_task(parsed_args)

  # load model
  model, args = model_load(parsed_args.model, task, model_overrides=eval(parsed_args.model_overrides))
  assert len(model) == 1  # don't support ensemble for the moment
  model = model[0]

  output_embed_dim = model.decoder.layers[-1].final_layer_norm.weight.size(0)
  for param in model.parameters():
    param.requires_grad = False
  model.decoder.adaptive_softmax = None
  filter_ = torch.nn.Parameter(torch.Tensor(2, output_embed_dim))
  torch.nn.init.normal_(filter_, mean=0, std=output_embed_dim ** -0.5)
  model.decoder.embed_out = filter_
  assert model.decoder.embed_out.requires_grad == True

  if use_cuda:
    model = model.cuda()

  if parsed_args.finetune_optimizer == "SGD":
    optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.decoder.parameters()), lr=parsed_args.finetune_learning_rate, momentum=parsed_args.finetune_momentum)
  else:
    optimizer = eval("torch.optim." + parsed_args.finetune_optimizer)(filter(lambda param: param.requires_grad, model.decoder.parameters()), lr=parsed_args.finetune_learning_rate)

  scheduler = ReduceLROnPlateau(optimizer, 'min', factor=parsed_args.finetune_lr_shrink)

  prefix_corpus_train = data.SentCorpus(os.path.join(parsed_args.data_prefix, "train.prefx.txt"), task.source_dictionary, append_eos=False)
  tag_corpus_train = data.read_tags(os.path.join(parsed_args.data_prefix, "train.tag.txt"))
  prefix_corpus_dev = data.SentCorpus(os.path.join(parsed_args.data_prefix, "valid.prefx.txt"), task.source_dictionary, append_eos=False)
  tag_corpus_dev = data.read_tags(os.path.join(parsed_args.data_prefix, "valid.tag.txt"))

  # batchify3: pads at the beginning, may introduce slight bias, but should be fine overall
  prefix_data_train = batchify2(prefix_corpus_train.test, parsed_args.finetune_batch_size, prefix_corpus_train.dictionary.pad_index)
  tag_data_train = batchify2(tag_corpus_train, parsed_args.finetune_batch_size, prefix_corpus_train.dictionary.pad_index)
  prefix_data_dev = batchify2(prefix_corpus_dev.test, parsed_args.finetune_batch_size, prefix_corpus_train.dictionary.pad_index)
  tag_data_dev = batchify2(tag_corpus_dev, parsed_args.finetune_batch_size, prefix_corpus_train.dictionary.pad_index)

  for epoch in range(parsed_args.max_epoch):
    model = finetune(prefix_data_train, tag_data_train, model, parsed_args.outdir, optimizer, epoch, parsed_args.cuda)
    valid_loss = validate(prefix_data_dev, tag_data_dev, model, parsed_args.cuda)
    scheduler.step(valid_loss)
    print("valid loss after epoch {0}: {1}".format(epoch, valid_loss))


def cli_main():
    parser = options.get_eval_lm_parser()
    parser.add_argument("--model", type=str, metavar="PATH", required=True, help="path to the saved model")
    parser.add_argument("--outdir", type=str, metavar="PATH", required=True, help="path to the finetuned model")
    parser.add_argument("--data-prefix", type=str, metavar="PATH", required=True, help="path to test data (without the .prefx.txt, .tag.txt, .subjs.txt suffix)")
    parser.add_argument("--finetune-batch-size", type=int, default=10, help="test batch size")
    parser.add_argument("--cuda", action='store_true', default=False, help="use cuda")

    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--finetune-optimizer", type=str, choices=["SGD", "Adam"], default="Adam")
    parser.add_argument("--finetune-momentum", type=float, default=0.99, help="momentum for SGD")
    parser.add_argument("--finetune-learning-rate", '-lr', type=float, default=0.001)
    parser.add_argument("--finetune-lr-shrink", type=float, default=0.5)

    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()

