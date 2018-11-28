import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from . import FairseqGenerator
from fairseq.modules import LearnedPositionalEmbedding, MultiheadAttention


class BottleneckLayer(nn.Module):

    def __init__(self, in_features, bn_features, out_features,
                 fc_in_builder=nn.Linear, fc_out_builder=nn.Linear):
        super().__init__()
        self.fc_in = fc_in_builder(in_features, bn_features)
        self.fc_out = fc_out_builder(bn_features, out_features)

    def forward(self, x):
        bottleneck = torch.nn.functional.relu(self.fc_in(x))
        return torch.nn.functional.relu(self.fc_out(bottleneck))


class RefinementLayer(nn.Module):

    def __init__(self, char_dictionary, fc_in_builder, fc_out_builder,
                 hidden_size, dropout,
                 char_embed, pos_embed,
                 bottleneck_layers=0, bottleneck_factor=2,
                 refinement_composition="cnn",
                 num_attn_heads=1):

        super(RefinementLayer, self).__init__()
        self.char_embed = char_embed
        self.pos_embed = pos_embed
        self.char_embed_dim = char_embed.embedding_dim
        self.pos_embed_dim = pos_embed.embedding_dim
        self.dropout = dropout
        self.num_embeddings = len(char_dictionary)

        self.composition = refinement_composition
        if refinement_composition == "attn":
            self.fc_in = CharGenLinear(hidden_size, self.char_embed_dim + self.pos_embed_dim, bias=False, dropout=dropout)
            self.fc_in_rev = CharGenLinear(self.char_embed_dim + self.pos_embed_dim, hidden_size, bias=False, dropout=dropout)
            self.fc_in_rev.weight = self.fc_in.weight.transpose(0, 1)  # tie weights
            self.attn = MultiheadAttention(self.char_embed_dim + self.pos_embed_dim, num_attn_heads)
        elif refinement_composition == "cnn":
            self.compact_fc_in_rev = CharGenLinear(self.char_embed_dim, hidden_size, bias=False, dropout=dropout)
        else:
            raise NotImplementedError

        bottlenecks = []
        for _ in range(bottleneck_layers):
            bottlenecks.append(BottleneckLayer(self.char_embed_dim + self.pos_embed_dim,
                                        (self.char_embed_dim + self.pos_embed_dim) // bottleneck_factor,
                                        self.char_embed_dim + self.pos_embed_dim,
                                        fc_in_builder,
                                        fc_out_builder))
        self.bottleneck = nn.Sequential(*tuple(bottlenecks))
        self.fc_out = fc_out_builder(self.char_embed_dim + self.pos_embed_dim, self.num_embeddings)

    def forward(self, logits, decoder_embed):
        """

        :param logits: (batch_size, max_seq_len, max_word_len, num_embeddings)
        :param decoder_embed: (batch_size, max_seq_len, hidden_dim)
        :return:
        """
        assert logits.size(-1) == self.num_embeddings

        batch_size, max_seq_len, max_word_len, _ = logits.size()

        pos_idx = torch.arange(max_word_len).type_as(logits).long()
        pos_idx = pos_idx.unsqueeze(0).expand(batch_size, max_seq_len, -1)  # (batch_size, max_seq_len, max_word_len)
        pos_embeds = self.pos_embed(pos_idx)  # (batch_size, max_seq_len, max_word_len, pos_embed_dim)
        pos_embeds = pos_embeds.view(batch_size * max_seq_len, max_word_len, -1).transpose(0, 1)

        logits = logits.view(-1, self.num_embeddings)
        # FIXME: get rid of next three lines -- forget about gumbel softmax for the moment
        # char_one_hot = F.gumbel_softmax(logits).view(batch_size, max_seq_len, max_word_len, self.num_embeddings)
        # char_embeds = char_one_hot.matmul(self.char_embed.weight)  # FIXME: make sure we are not using fancier embedding retrieval

        char_idx = torch.argmax(logits, -1).view(batch_size, max_seq_len, max_word_len)  # no need for softmax
        char_idx.detach_()  # make sure gradient does not flow from here

        # ----------- composition options start -----------

        if self.composition == "attn":
            # composition option 1: attn_compose
            comp_embeds = self.attn_composition(char_idx, pos_embeds, decoder_embed)

        elif self.composition == "cnn":
            # cmposition option 2: cnn_compose
            comp_embeds = self.cnn_compose(char_idx, decoder_embed)

        else:
            raise NotImplementedError

        # ----------- composition options end -----------

        # (batch_size, max_seq_len, max_word_len hidden_size)
        comp_embeds = comp_embeds.unsqueeze(1).expand(-1, max_word_len, -1)
        bn_in = torch.cat(
            (comp_embeds, pos_embeds.transpose_(0, 1)),
            dim=-1)

        bn_out = self.bottleneck(bn_in)
        ret_logit = self.fc_out(bn_out).view(batch_size, max_seq_len, max_word_len, -1)  # (batch_size, max_seq_len, max_word_len, num_embeddings)
        return ret_logit

    def attn_composition(self, char_idx, pos_embeds, decoder_embed):
        batch_size, max_seq_len, max_word_len = char_idx.size()
        char_embeds = F.embedding(char_idx, self.char_embed.weight, padding_idx=self.char_embed.padding_idx)  # don't need composition here
        char_embeds = char_embeds.view(batch_size * max_seq_len, max_word_len, -1).transpose(0, 1)

        kv_embeds = torch.cat((char_embeds, pos_embeds), dim=-1)
        q_embeds = self.fc_in(decoder_embed)
        q_embeds = F.dropout(q_embeds, p=self.dropout, training=self.training)
        q_embeds = q_embeds.view(batch_size * max_seq_len, -1)
        q_embeds = q_embeds.unsqueeze(0)  # (1, batch_size * max_seq_len, attn_embed_dim)
        attn_embeds, _ = self.attn(q_embeds, kv_embeds, kv_embeds)
        attn_embeds.squeeze_()
        attn_embeds = self.fc_in_rev(attn_embeds)  # (batch_size * max_seq_len, hidden_size)
        attn_embeds = F.dropout(attn_embeds, p=self.dropout, training=self.training)
        return attn_embeds

    def cnn_composition(self, char_idx, decoder_embed):
        batch_size, max_seq_len, max_word_len = char_idx.size()
        cnn_embeds = self.char_embed(char_idx)  # (batch_size, max_seq_len, self.num_embeddings)
        cnn_embeds = self.compact_fc_in_rev(cnn_embeds) + decoder_embed
        cnn_embeds = cnn_embeds.view(batch_size * max_seq_len, -1)  # (batch_size * max_seq_len, hidden_size)
        return cnn_embeds


class NonAutoRegCharGenerator(FairseqGenerator):

    def __init__(self, char_dictionary, fc_in_builder, fc_out_builder, hidden_size,
                 char_embed, pos_embed_dim, dropout, max_word_len,
                 input_embed=None, always_have_fc_in=False,
                 use_decoder_highway=False,
                 bottleneck_layers=0, bottleneck_factor=4,
                 refinement_layers=0, tie_refinements=False,
                 refinement_composition="cnn",
                 refinement_autoenc=False,
                 length_prediction=False,
                ):
        """
        A basic non-autoregressive character generator, without length prediction

        :param char_dictionary:
        :param fc_in_builder:
        :param fc_out_builder:
        :param hidden_size:
            TODO: We keep this for now, but chances are that this would be deleted in the future as the only place
                this is really used is for adaptive softmax in lstm. Especially, note that for transformer
                hidden_size == embed_dim.
        :param char_embed:
        :param pos_embed_dim:
        :param dropout:
        :param max_word_len: the maximum length of a word, in terms of characters
        :param input_embed:
            TODO: unused in the original code
        :param always_have_fc_in:
        :param use_decoder_highway:
        :param bottleneck_layers:
        :param bottleneck_factor:
        :param refinement_layers:
        :param tie_refinements:
        """

        assert len(char_dictionary) == char_embed.num_embeddings
        num_embeddings = len(char_dictionary)
        super(NonAutoRegCharGenerator, self).__init__(num_embeddings)

        self.dropout = dropout
        self.max_word_len = max_word_len
        self.fc_in = None

        char_embed_dim = char_embed.embedding_dim
        self.char_embed = nn.Embedding(num_embeddings, char_embed_dim, padding_idx=char_embed.padding_idx)
        self.char_embed.weight = char_embed.weight  # tie the weight

        # note this positional embedding shouldn't be shared with the word-level positional embedding in, e.g., fconv
        # self.pos_embed = PositionalEmbedding(max_word_len, pos_embed_dim, char_dictionary.pad(), left_pad)

        # also, what the fairseq positional embedding does is not quite a good fit for what we are doing
        self.pos_embed = nn.Embedding(self.max_word_len, pos_embed_dim)

        if hidden_size != char_embed_dim or always_have_fc_in:
            self.fc_in = fc_in_builder(hidden_size, char_embed_dim, dropout=dropout)

        if input_embed:
            self.fc_out = nn.Linear(char_embed_dim + pos_embed_dim, num_embeddings, bias=False)
            assert(char_embed_dim == input_embed.weight.size(0))
            assert(num_embeddings == input_embed.weight.size(1))
            self.fc_out.weight = input_embed.weight
        else:
            self.fc_out = fc_out_builder(char_embed_dim + pos_embed_dim, num_embeddings)

        bottlenecks = []
        for _ in range(bottleneck_layers):
            bottlenecks.append(BottleneckLayer(char_embed_dim + pos_embed_dim,
                                        (char_embed_dim + pos_embed_dim) // bottleneck_factor,
                                        char_embed_dim + pos_embed_dim,
                                        fc_in_builder,
                                        fc_out_builder))
        if bottlenecks:
            self.bottleneck = nn.Sequential(*tuple(bottlenecks))
        else:
            self.bottleneck = None

        self.refinement_layers = refinement_layers
        self.tie_refinements = tie_refinements
        if tie_refinements:
            self.refinement = RefinementLayer(char_dictionary, fc_in_builder, fc_out_builder,
                                              hidden_size, dropout, self.char_embed, self.pos_embed,
                                              bottleneck_layers, bottleneck_factor,
                                              refinement_composition)
        else:
            refinements = []
            for _ in range(refinement_layers):
                refinements.append(RefinementLayer(char_dictionary, fc_in_builder, fc_out_builder,
                                              hidden_size, dropout, self.char_embed, self.pos_embed,
                                              bottleneck_layers, bottleneck_factor,
                                              refinement_composition))

            if refinements:
                self.refinement = nn.ModuleList(refinements)
            else:
                self.refinement = None

        if refinement_autoenc:
            self.refinement_autoenc = RefinementAutoEncoder(self.refinement)
        elif not tie_refinements:
            raise AttributeError("Refinement autoenc loss has to be used with tied refinement layers.")
        else:
            self.refinement_autoenc = None

        if length_prediction and tie_refinements:
            self.length_predictor = LengthPredictionGenerator(hidden_size, max_word_len)
        else:
            self.length_predictor = None


    def forward(self, x, log_probs, sample=None):
        """

        :param x: of size (batch_size, max_seq_len, hidden_dim)
        :param log_probs: boolean indicating whether to output log prob or real prob
        :param sample: when scoring a sample, the sample should be passed here
        :return:
        """
        decoder_embed = x.clone()
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

        if self.bottleneck:
            x = self.bottleneck(x)

        x = self.fc_out(x)  # (batch_size, max_seq_len, max_word_len, num_embeddings)

        if self.refinement:
            if self.tie_refinements:
                for _ in range(self.refinement_layers):
                    x = self.refinement(x, decoder_embed)
            else:
                for ref in self.refinement:
                    x = ref(x, decoder_embed)

        return self.get_normalized_probs(x, log_probs, sample)


class LengthPredictionGenerator(nn.Module):

    def __init__(self, hidden_size, max_len):
        super(LengthPredictionGenerator, self).__init__(self)
        self.linear = CharGenLinear(hidden_size, max_len)

    def forward(self, decoder_embed):
        return torch.nn.functional.softmax(
            torch.nn.functional.relu(self.linear(decoder_embed)), dim=-1)


class RefinementAutoEncoder(nn.Module):

    def __init__(self, refinement_layer):
        super(RefinementAutoEncoder, self).__init__(self)
        self.refinement_layer = refinement_layer

    def forward(self, noisy_char_idx, decoder_embed):
        batch_size, max_seq_len, max_word_len = noisy_char_idx.size()

        pos_idx = torch.arange(max_word_len).type_as(noisy_char_idx)
        pos_idx = pos_idx.unsqueeze(0).expand(batch_size, max_seq_len, -1)  # (batch_size, max_seq_len, max_word_len)
        pos_embeds = self.refinement_layer.pos_embed(pos_idx)  # (batch_size, max_seq_len, max_word_len, pos_embed_dim)
        pos_embeds = pos_embeds.view(batch_size * max_seq_len, max_word_len, -1).transpose(0, 1)

        # ----------- composition options start -----------

        if self.composition == "attn":
            # composition option 1: attn_compose
            comp_embeds = self.refinement_layer.attn_composition(noisy_char_idx, pos_embeds, decoder_embed)

        elif self.composition == "cnn":
            # cmposition option 2: cnn_compose
            comp_embeds = self.cnn_compose(noisy_char_idx, decoder_embed)

        else:
            raise NotImplementedError

        # ----------- composition options end -----------

        # (batch_size, max_seq_len, max_word_len hidden_size)
        comp_embeds = comp_embeds.unsqueeze(1).expand(-1, max_word_len, -1)
        bn_in = torch.cat(
            (comp_embeds, pos_embeds.transpose_(0, 1)),
            dim=-1)

        bn_out = self.bottleneck(bn_in)
        logit = self.fc_out(bn_out).view(batch_size, max_seq_len, max_word_len, -1)  # (batch_size, max_seq_len, max_word_len, num_embeddings)
        return torch.nn.functional.softmax(logit, dim=-1)


# copied from fconv
def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

# copied from fconv and modified
def CharGenLinear(in_features, out_features, dropout=0, bias=True):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    if bias:
        nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)
