# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-09-06
#
# Distributed under terms of the MIT license.

import torch
from torch import nn
import pdb

from salience import SalienceType, SalienceManager

class AdaptiveInputWithSalience(nn.Module):
  """
  A wrapper over the fairseq AdaptiveInput class to compute saliency.

  Note that because the original forward function in AdaptiveInput class will apply the
  linear projection to make the look-up results from different bins of same dimension,
  we'll have to ditch that function and re-implement `forward`, so the collected gradients
  will be the ones that propagated through these linear projections.
  """

  def __init__(self, adaptive_input, salience_type=None,
               smooth_factor=0.15, smooth_samples=30,  # for smoothgrad
               integral_steps=100):

    super().__init__()
    self.embedding = adaptive_input
    self.salience_type = salience_type
    self.smooth_factor = smooth_factor
    self.smooth_samples = smooth_samples
    self.integral_steps = integral_steps
    self.activated = False


  def activate(self, salience_type):
    """
    Salience should not be computed for all evaluations, like validation during training.
    In these cases, SalienceEmbedding should act the same way as normal word embedding.
    This switch should be turned on when salience needs to be computed.
    """
    self.activated = True
    self.salience_type = salience_type


  def deactivate(self):
    self.activated = False


  def adaptive_lookup(self, input):
    """
    Look up the embeddings in an adaptive manner, and return all the embeddings
    that's been queried in a list.

    Note: all the returned embeddings will be flat when returned, to recover their shape, use masks.
    """

    embs = []
    masks = []
    for i in range(len(self.embedding.cutoff)):
      # figure out mask
      mask = input.lt(self.embedding.cutoff[i])
      if i > 0:
        mask.mul_(input.ge(self.embedding.cutoff[i - 1]))
        chunk_input = input[mask] - self.embedding.cutoff[i - 1]
        masks.append(mask)
      else:
        chunk_input = input[mask]
        masks.append(mask)

      # if there are unmasked entries, do the query
      # otherwise, insert a placeholder
      if mask.any():
        embs.append(self.embedding.embeddings[i][0](chunk_input))
      else:
        embs.append(None)

    assert len(embs) == len(masks)
    return embs, masks


  def project_adaptive_lookup_results(self, input, embs, masks):
    """
    Apply the linear projections in the original forward function to make all returned
    embeddings of same dimension.
    """

    result = self.embedding._float_tensor.new(input.shape + (self.embedding.embedding_dim,))
    for i in range(len(self.embedding.cutoff)):
      if embs[i] is not None:
        result[masks[i]] = self.embedding.embeddings[i][1](embs[i])  # apply linear projection
    # ret = torch.stack(result, dim=0)  # needs to check shape for this one
    return result


  def forward(self, input, batch_first=True):
    """
    Can alternate with batch_first and batch_second.
    """

    if self.salience_type and self.activated:
      # in case where multiple samples are needed
      # repeat the samples and set accompanying parameters accordingly
      orig_size = list(input.size())
      new_size = orig_size
      new_size_expand = None
      if self.salience_type == SalienceType.smoothed:
        new_size_expand = tuple([ orig_size[0], orig_size[1], self.smooth_samples ] + orig_size[2:]) \
                            if not batch_first \
                            else tuple([ orig_size[0], self.smooth_samples ] + orig_size[1:])

      elif self.salience_type == SalienceType.integral:
        new_size_expand = tuple([ orig_size[0], orig_size[1], self.integral_steps ] + orig_size[2:]) \
                            if not batch_first \
                            else tuple([ orig_size[0], self.integral_steps ] + orig_size[1:])

      if new_size_expand and not batch_first:
        new_size = tuple([new_size_expand[0], new_size_expand[1] * new_size_expand[2]] + list(new_size_expand[3:]))
        input = input.unsqueeze(2).expand(*new_size_expand).contiguous().view(*new_size)
      elif new_size_expand:
        new_size = tuple([new_size_expand[0] * new_size_expand[1]] + list(new_size_expand[2:]))
        input = input.unsqueeze(1).expand(*new_size_expand).contiguous().view(*new_size)

    # normal embedding query
    embs, masks = self.adaptive_lookup(input)
    sel = torch.ones_like(input).float()
    sel.requires_grad = True
    sel.register_hook(lambda grad: SalienceManager.compute_salience(grad))

    if self.salience_type and self.activated:
      # iterate through the bins
      for i in range(len(self.embedding.cutoff)):
        x = embs[i]  # embedding queried with this bin, flattened
        mask = masks[i]

        # none of the words are queried in this bin, continue
        if not mask.any():
          continue

        masked_sel = torch.masked_select(sel, mask)

        if not (self.salience_type == SalienceType.li or \
              self.salience_type == SalienceType.li_smoothed):
          if self.salience_type == SalienceType.integral:
            alpha = torch.arange(0, 1, 1 / self.integral_steps) + 1 / self.integral_steps  # (0, 1] rather than [0, 1)
            alpha = alpha.unsqueeze(0).expand(torch.sum(mask[0]), self.integral_steps).contiguous().view(-1)  # make its shape on first dimension the same as x (the flattened embedding)
            alpha = alpha.type_as(x)
            x = (x * masked_sel.unsqueeze(1) * alpha.unsqueeze(1)) if batch_first else (x * masked_sel * alpha)
          else:
            x = (x * masked_sel.unsqueeze(1)) if batch_first else (x * masked_sel)
        else:
            raise NotImplementedError

        if (self.salience_type == SalienceType.smoothed or \
              self.salience_type == SalienceType.li_smoothed) and \
              self.smooth_factor > 0.0:
            x = x + torch.normal(torch.zeros_like(x), \
                    torch.ones_like(x) * self.smooth_factor * (torch.max(x) - torch.min(x)))

        embs[i] = x

    ret = self.project_adaptive_lookup_results(input, embs, masks)
    return ret

