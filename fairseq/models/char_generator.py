import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from . import FairseqGenerator
from fairseq.modules import LearnedPositionalEmbedding


class Denoisier(nn.Module):

    def __init__(self, in_features, bn_features, out_features):
        super().__init__()
        self.fc_in = nn.Linear(in_features, bn_features)
        self.fc_out = nn.Linear(bn_features, out_features)

    def forward(self, x):
        bottleneck = torch.nn.functional.relu(self.fc_in(x))
        return torch.nn.functional.relu(self.fc_out(bottleneck))


class NonAutoRegCharGenerator(FairseqGenerator):

    def __init__(self, char_dictionary, fc_in_builder, fc_out_builder, hidden_size,
                 char_embed_dim, pos_embed_dim, dropout, max_word_len,
                 input_embed=None, always_have_fc_in=False,
                 denoisier_layers=0, denoisier_bottleneck_factor=2):
        """
        A basic non-autoregressive character generator, without length prediction

        :param char_dictionary:
        :param fc_in_builder:
        :param hidden_size:
            TODO: We keep this for now, but chances are that this would be deleted in the future as the only place
                this is really used is for adaptive softmax in lstm. Especially, note that for transformer
                hidden_size == embed_dim.
        :param char_embed_dim:
        :param pos_embed_dim:
        :param dropout:
        :param max_word_len: the maximum length of a word, in terms of characters
        :param input_embed:
            TODO: unused in the original code
        :param always_have_fc_in:
        :param denoisier_layers:
        :param denoisier_bottleneck_factor:
        """
        num_embeddings = len(char_dictionary)
        super(NonAutoRegCharGenerator, self).__init__(num_embeddings)

        self.dropout = dropout
        self.max_word_len = max_word_len
        self.fc_in = None
        if hidden_size != char_embed_dim or always_have_fc_in:
            self.fc_in = fc_in_builder(hidden_size, char_embed_dim)

        if input_embed:
            self.fc_out = nn.Linear(char_embed_dim + pos_embed_dim, num_embeddings, bias=False)
            assert(char_embed_dim == input_embed.weight.size(0))
            assert(num_embeddings == input_embed.weight.size(1))
            self.fc_out.weight = input_embed.weight
        else:
            self.fc_out = fc_out_builder(char_embed_dim + pos_embed_dim, num_embeddings, dropout=dropout)

        # note this positional embedding shouldn't be shared with the word-level positional embedding in, e.g., fconv
        # self.pos_embed = PositionalEmbedding(max_word_len, pos_embed_dim, char_dictionary.pad(), left_pad)

        # also, what the fairseq positional embedding does is not quite a good fit for what we are doing
        self.pos_embed = nn.Embedding(self.max_word_len, pos_embed_dim)

        denoisiers = []
        for _ in range(denoisier_layers):
            denoisiers.append(Denoisier(char_embed_dim + pos_embed_dim,
                                        (char_embed_dim + pos_embed_dim) // denoisier_bottleneck_factor,
                                        char_embed_dim + pos_embed_dim))
        if denoisiers:
            self.denoisier = nn.Sequential(*tuple(denoisiers))
        else:
            self.denoisier = None

    def forward(self, x, log_probs, sample=None):
        """

        :param x: of size (batch_size, max_seq_len, hidden_dim)
        :param log_probs: boolean indicating whether to output log prob or real prob
        :param sample: when scoring a sample, the sample should be passed here
        :return:
        """
        if self.fc_in is not None:
            x = torch.nn.functional.relu(self.fc_in(x))
            x = F.dropout(x, p=self.dropout, training=self.training)  # TODO: transformer doesn't seem to be using this
        batch_size = x.size(0)
        max_seq_len = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, self.max_word_len, -1)  # (batch_size, max_seq_len, max_word_len, hidden_dim)
        pos_idx = torch.arange(self.max_word_len).type_as(x).long()
        pos_idx = pos_idx.unsqueeze(0).expand(batch_size, max_seq_len, -1)  # (batch_size, max_seq_len, max_word_len)
        pos = self.pos_embed(pos_idx)  # (batch_size, max_seq_len, max_word_len, pos_embed_dim)
        x = torch.cat((x, pos), dim=3)  # (batch_size, max_seq_len, max_word_len, hidden_dim + pos_embed_dim)

        if self.denoisier:
            x = self.denoisier(x)

        x = self.fc_out(x)  # (batch_size, max_seq_len, max_word_len, num_embeddings)
        return self.get_normalized_probs(x, log_probs, sample)


# copied from fconv
def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
