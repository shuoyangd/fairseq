# -*- coding: utf-8 -*-
#
# Copyright © 2021 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2021-01-18
#
# Distributed under terms of the MIT license.

from fairseq import checkpoint_utils, options, tasks, utils
from train_estimator import ConfidenceRegressorWithModel
import logging
import pdb
import torch

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = options.get_generation_parser()
group = opt_parser.add_argument_group("Confidence Model Inference")
group.add_argument("--model", required=True, metavar="PATH", type=str,
                        help="path to the translation model")
group.add_argument("--gpu", action="store_true", default=False,
                        help="if gpu training should be used, set this option")

def main(options):
  # prepare_dataset
  task = tasks.setup_task(options)
  task.load_dataset("test", append_eos_to_target=True)
  test_dataset = task.datasets["test"]

  # build model
  logging.info("loading transformer checkpoint...")
  estimator = torch.load(options.model, map_location=lambda storage, loc: storage)
  if options.gpu:
    estimator = estimator.cuda()

  batch_itr = task.get_batch_iterator(
    dataset=test_dataset,
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
  epoch_itr = batch_itr.next_epoch_itr(shuffle=False)

  options.score_reference = True  # hard set to true, we assume a scorer is used from now on
  generator = task.build_generator(
    None, options,
  )  # first arg not going to be used anyway since we are building a scorer

  n_updates = 0
  for sample in epoch_itr:
    bsz = sample["target"].size(0)
    sample = utils.move_to_cuda(sample) if options.gpu else sample
    with torch.no_grad():
      y = estimator(generator, sample).squeeze().tolist()
      for elem in y:
        print(elem)


if __name__ == "__main__":
  args = options.parse_args_and_arch(opt_parser)
  main(args)