# some simple test for character generator

from fairseq.models import NonAutoRegCharGenerator
from fairseq.data.dictionary import Dictionary

import torch
from torch import nn

import math

dict = Dictionary()
for char in "abcdefghijklmnopqrstuvwxyz ":
  dict.add_symbol(char)

# copied from fconv
def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)

g = NonAutoRegCharGenerator(
    char_dictionary=dict,
    fc_in_builder=Linear,
    fc_out_builder=Linear,
    hidden_size=512,
    char_embed_dim=17,
    pos_embed_dim=13,
    dropout=0.1,
    max_word_len=15,
    denoisier_layers=4,
)

x = torch.Tensor(31, 128, 512)
o = g(x, False)
