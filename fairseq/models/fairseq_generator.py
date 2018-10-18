# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    AdaptiveSoftmax,
)


class FairseqGenerator(nn.Module):
    """Base class for generators."""

    def __init__(self, num_embeddings):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, decoder_output, log_probs):
        raise NotImplementedError

    def get_normalized_probs(self, logits, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            assert sample is not None and 'target' in sample
            out = self.adaptive_softmax.get_log_prob(logits, sample['target'])
            return out.exp_() if not log_probs else out

        logits = logits.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


class BasicFairseqGenerator(FairseqGenerator):

    def __init__(self, dictionary, linear_builder, hidden_size, embed_dim, out_embed_dim,
                 dropout, input_embed=None, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0):
        """
        A unified generator for lstm, fconv and transformer.

        TODO: not sure what input dimension should I use for Adaptive Softmax.. transformer uses output_embed_dim,
        lstm uses (input) embed_dim, and fconv uses in_channels (hidden_size). we are using hidden_size for the moment.

        :param dictionary:
        :param linear_builder:
        :param hidden_size:
        :param embed_dim:
            TODO: We keep this for now, but chances are that this would be deleted in the future as the only place
                this is really used is for adaptive softmax in lstm. Especially, note that for transformer
                hidden_size == embed_dim.
        :param out_embed_dim:
        :param dropout:
        :param input_embed:
        :param adaptive_softmax_cutoff:
        :param adaptive_softmax_dropout:
            TODO: unused in the original code
        """
        num_embeddings = len(dictionary)
        super(BasicFairseqGenerator, self).__init__(num_embeddings)

        self.dropout = dropout
        self.adaptive_softmax = None
        if adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
                                                    dropout=dropout)
        else:
            if hidden_size != out_embed_dim:
                self.fc_in = linear_builder(hidden_size, out_embed_dim)

            self.fc_out = linear_builder(out_embed_dim, num_embeddings, dropout=dropout)
            if input_embed:
                assert(out_embed_dim == input_embed.weight.size(0))
                assert(num_embeddings == input_embed.weight.size(1))
                self.fc_out.weight = input_embed.weight

    def forward(self, x, log_probs, sample=None):
        if self.adaptive_softmax is None:
            if self.fc_in is not None:
                x = self.fc_in(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)  # TODO: transformer doesn't seem to be using this
            x = self.fc_out(x)
        else:
            # TODO: adaptive softmax is not yet used here
            raise NotImplementedError
        return self.get_normalized_probs(x, log_probs, sample)
