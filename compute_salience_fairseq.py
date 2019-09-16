# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-05-01
#
# Distributed under terms of the MIT license.

import argparse
import data
import logging
import torch
import torch.nn.functional as F
# cudnn backward cannot be called at eval mode
import torch.backends.cudnn as cudnn
cudnn.enabled = False

from fairseq import options, progress_bar, tasks, utils
from fairseq.utils import import_user_module
import pdb

from utils import batchify2
from salience import SalienceManager
from salience import SalienceType
from fairseq_salience import AdaptiveInputWithSalience

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

"""
opt_parser = argparse.ArgumentParser(description="compute salience from fairseq language model")
opt_parser.add_argument("--save", type=str, metavar="PATH", required=True, help="path to the saved model")
opt_parser.add_argument("--data-prefix", type=str, metavar="PATH", required=True, help="path to test data (without the .prefx.txt, .tag.txt, .subjs.txt suffix)")
opt_parser.add_argument("--output", type=str, metavar="PATH", required=True, help="path to output file")
opt_parser.add_argument("--batch-size", type=int, default=10, help="test batch size")
opt_parser.add_argument("--salience-type", type=str, choices=["vanilla", "smoothed", "integral", "li", "li_smoothed"], default="vanilla", help="type of salience")
opt_parser.add_argument("--cuda", action='store_true', default=False, help="use cuda")
"""


def model_load(fn):
  with open(fn, 'rb') as f:
    # model, criterion, optimizer = torch.load(f, map_location=lambda storage, loc: storage)
    model = torch.load(f, map_location=lambda storage, loc: storage)
    return model


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


def evaluate(prefix_data, tag_data, subjs_data, model, outdir, cuda=False):
  total_count = 0
  match_count = 0
  pos_count = 0
  neg_count = 0
  pos_match_count = 0
  neg_match_count = 0
  verum_salience_output = open(outdir + ".verum", 'w')
  malum_salience_output = open(outdir + ".malum", 'w')
  pos_index_output = open(outdir + ".pos.idx", 'w')
  passed_index_output = open(outdir + ".passed.idx", 'w')
  # bsz and batch_size are the same thing :)
  for (prefix_batch, prefix_mask), (tag_batch, _), (subjs_batch, subjs_mask) in \
      zip(zip(prefix_data[0], prefix_data[1]), zip(tag_data[0], tag_data[1]), zip(subjs_data[0], subjs_data[1])):
    batch_size = subjs_batch.size(1)
    sample_size = 1
    if model.decoder.embed_tokens.salience_type == SalienceType.smoothed:
      sample_size = model.decoder.embed_tokens.smooth_samples
    elif model.decoder.embed_tokens.salience_type == SalienceType.integral:
      sample_size = model.decoder.embed_tokens.integral_steps
    bsz_samples = batch_size * sample_size

    padded_seq_len = prefix_batch.size(0)
    # hidden = model.init_hidden(bsz_samples)
    # no_salience_hidden = model.init_hidden(batch_size)
    # XXX: note, these batch are not batch_first, but the model requires a batch_first input
    if cuda:
      prefix_batch = prefix_batch.cuda()  # (src_len, bsz)  # TODO may need some expanding
      tag_batch = tag_batch.cuda()
      prefix_mask = prefix_mask.cuda()
      subjs_batch = subjs_batch.cuda()  # (2, bsz) XXX: shouldn't be expanded

    # first evaluation: get probability of two tag
    real_salience_type = model.decoder.embed_tokens.salience_type
    model.decoder.embed_tokens.salience_type = None
    # model.reset()  # this is for QRNN, don't need this for transformer
    model.decoder.embed_tokens.deactivate()
    no_salience_output, _ = model(prefix_batch.transpose(0, 1), None)  # (batch_size, padded_seq_len, vocab_size)
    no_salience_probs = model.get_normalized_probs([no_salience_output], log_probs=True).transpose(0, 1)  # (padded_seq_len, batch_size, vocab_size)

    # second evaluation: salience computation
    # model.reset()  # this is for QRNN, don't need this for transformer
    model.decoder.embed_tokens.activate(real_salience_type)
    output, _ = model(prefix_batch.transpose(0, 1), None)  # (batch_size, padded_seq_len, vocab_size)
    probs = model.get_normalized_probs([output], log_probs=True).transpose(0, 1)  # (padded_seq_len, batch_size, vocab_size)

    final_prefix_index = torch.sum(prefix_mask, dim=0).unsqueeze(0) - 1  # (1, bsz)
    final_prefix_index = final_prefix_index.unsqueeze(2).expand(-1, -1, probs.size(-1))  # (1, batch_size, vocab_size)
    no_salience_verb_probs = torch.gather(no_salience_probs, 0, final_prefix_index).squeeze(0)  # (bsz, vocab_size)
    # this is the probability conditioned on ALL the words in the prefix, i.e., the probability of the verb
    final_prefix_index = final_prefix_index.unsqueeze(2).expand(-1, -1, sample_size, -1).contiguous().view(1, bsz_samples, -1)  # (1, batch_size * sample_size, vocab_size)
    verb_probs = torch.gather(probs, 0, final_prefix_index).squeeze(0)  # (bsz * n_samples, vocab_size)

    SalienceManager.backward_fairseq_with_salience_single_timestep(verb_probs, tag_batch[0, :], model)
    averaged_salience_verum = SalienceManager.average_single_timestep(batch_first=True)  # (src_len, bsz)
    SalienceManager.clear_salience()
    SalienceManager.backward_fairseq_with_salience_single_timestep(verb_probs, 1 - tag_batch[0], model)
    averaged_salience_malum = SalienceManager.average_single_timestep(batch_first=True)  # (src_len, bsz)
    SalienceManager.clear_salience()

    verum_salience_output.write(str(averaged_salience_verum.squeeze().tolist()) + "\n")
    malum_salience_output.write(str(averaged_salience_malum.squeeze().tolist()) + "\n")

    # XXX: must use no salience verb probs here
    # salience methods that involve samples will change the verb probability here
    # causing small variance in the test applied
    verum_probs = torch.gather(no_salience_verb_probs, -1, tag_batch.transpose(0, 1))  # (bsz, n_samples)
    malum_probs = torch.gather(no_salience_verb_probs, -1, 1 - tag_batch.transpose(0, 1))  # (bsz, n_samples)
    # verum_probs = torch.mean(verum_probs, dim=1)  # (bsz,)
    # malum_probs = torch.mean(malum_probs, dim=1)  # (bsz,)
    is_pos = (verum_probs > malum_probs)
    is_neg = (verum_probs < malum_probs)
    # is_pos = (verum_probs < malum_probs)
    # is_neg = (verum_probs > malum_probs)

    for i in range(batch_size):
        subject = subjs_batch[0, i]  # scalar
        attractors = subjs_batch[1:, i]  # (number of attractors, )
        attractors_mask = subjs_mask[1:, i]  # (number of attractors, )
        noa = torch.sum(attractors_mask)  # noa -> number of attractors

        pos_subj_salience = averaged_salience_verum[i, subject]
        neg_subj_salience = averaged_salience_malum[i, subject]
        pos_subj_salience = pos_subj_salience.expand(noa)  # (number of attractors,)
        neg_subj_salience = neg_subj_salience.expand(noa)  # (number of attractors,)
        attractors = attractors[attractors_mask]
        pos_attr_salience = torch.gather(averaged_salience_verum[i, :], 0, attractors).squeeze()  # (number of attractors,)
        neg_attr_salience = torch.gather(averaged_salience_malum[i, :], 0, attractors).squeeze()  # (number of attractors,)

        if torch.all(pos_subj_salience > pos_attr_salience) and is_pos[i]:
          pos_match_count += 1
          match_count += 1
          passed_index_output.write(str(total_count + i) + "\n")
        elif torch.any(neg_subj_salience < neg_attr_salience) and is_neg[i]:
          neg_match_count += 1
          match_count += 1
          passed_index_output.write(str(total_count + i) + "\n")

        if is_pos[i]:
          pos_index_output.write(str(total_count + i) + "\n")

    pos_count += torch.sum(is_pos)
    neg_count += torch.sum(is_neg)
    total_count += subjs_batch.size(1)
    logging.info("running frac: {0} / {1} = {2}".format(match_count, total_count, match_count / total_count))
    if pos_count != 0:
        logging.info("running pos frac: {0} / {1} = {2}".format(pos_match_count, pos_count, pos_match_count / pos_count.item()))
    else:
        logging.info("running pos frac: 0 / 0 = 0")
    if neg_count != 0:
        logging.info("running neg frac: {0} / {1} = {2}".format(neg_match_count, neg_count, neg_match_count / neg_count.item()))
    else:
        logging.info("running neg frac: 0 / 0 = 0")

  verum_salience_output.close()
  malum_salience_output.close()
  passed_index_output.close()
  pos_index_output.close()
  return (match_count, total_count, match_count / total_count, \
          pos_match_count, pos_count.item(), pos_match_count / pos_count.item(), \
          neg_match_count, neg_count.item(), neg_match_count / neg_count.item())


def main(parsed_args):
  assert parsed_args.path is not None, '--path required for evaluation!'
  torch.autograd.set_detect_anomaly(True)  # TODO: debug in-place

  import_user_module(parsed_args)
  print(parsed_args)
  use_cuda = torch.cuda.is_available() and not parsed_args.cpu
  task = tasks.setup_task(parsed_args)

  # load model
  # model, args = model_load(parsed_args.path, task, model_overrides=eval(parsed_args.model_overrides))
  model = model_load(parsed_args.path)
  for param in model.parameters():
    param.requires_grad = True
  # assert len(model) == 1  # don't support ensemble for the moment
  # model = model[0]
  if use_cuda:
    model = model.cuda()

  # wrap embedding so it can compute saliency
  model.decoder.embed_tokens = AdaptiveInputWithSalience(model.decoder.embed_tokens)
  # disable gradient computation
  model.eval()
  # enable gradient computation for embedding with salience
  model.decoder.embed_tokens.activate(eval("SalienceType." + parsed_args.salience_type))

  # read data
  prefix_corpus = data.SentCorpus(parsed_args.data_prefix + ".prefx.txt", task.source_dictionary, append_eos=False)
  tag_corpus = data.read_tags(parsed_args.data_prefix + ".tag.txt")
  subjs_corpus = data.read_subjs_data(parsed_args.data_prefix + ".subjs.txt")
  prefix_data = batchify2(prefix_corpus.test, parsed_args.salience_batch_size, prefix_corpus.dictionary.pad_index)
  tag_data = batchify2(tag_corpus, parsed_args.salience_batch_size, prefix_corpus.dictionary.pad_index)
  subjs_data = batchify2(subjs_corpus, parsed_args.salience_batch_size, -1)  # there won't be pad for this
  frac = evaluate(prefix_data, tag_data, subjs_data, model, parsed_args.output, parsed_args.cuda)
  print(frac)


def cli_main():
    parser = options.get_eval_lm_parser()
    parser.add_argument("--data-prefix", type=str, metavar="PATH", required=True, help="path to test data (without the .prefx.txt, .tag.txt, .subjs.txt suffix)")
    parser.add_argument("--salience-type", type=str, choices=["vanilla", "smoothed", "integral", "li", "li_smoothed"], default="vanilla", help="type of salience")
    parser.add_argument("--output", type=str, metavar="PATH", required=True, help="path to output file")
    parser.add_argument("--salience-batch-size", type=int, default=10, help="test batch size")
    parser.add_argument("--cuda", action='store_true', default=False, help="use cuda")
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
