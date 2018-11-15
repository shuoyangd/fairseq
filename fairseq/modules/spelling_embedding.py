import torch
import torch.nn as nn

from torch.nn import functional as F
from .highway import Highway

# A lower-level max pooling compared to PyTorch
# which essentially take max over some dimension
class MaxPool1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        return torch.max(tensor, dim=self.dim)[0]


# the class performing the action of composing the chracter embeddings into word embedding
class SpellingComposer(nn.Module):

    def __init__(self, char_emb_size, word_emb_size):
        super().__init__()
        self.char_emb_size = char_emb_size
        self.word_emb_size = word_emb_size

    def forward(self, char_emb):
        raise NotImplementedError


class CNNComposer(SpellingComposer):

    def __init__(self, char_emb_size, word_emb_size,
                 cnn_mix = 'cat', kernals="3456"):
        super().__init__(char_emb_size, word_emb_size)
        kernals = [int(i) for i in kernals]
        if cnn_mix == 'cat':
            assert word_emb_size % len(kernals) == 0
        cnns = []
        for k in kernals:
            if cnn_mix == 'cat':
                seq = nn.Sequential(
                    nn.Conv1d(char_emb_size, word_emb_size // len(kernals), k, padding=0),
                    nn.Tanh(),
                    MaxPool1d(-1)
                    )
            elif cnn_mix == 'add':
                seq = nn.Sequential(
                    nn.Conv1d(char_emb_size, word_emb_size, k, padding=0),
                    nn.Tanh(),
                    MaxPool1d(-1)
                    )
            else:
                raise NotImplementedError
            cnns.append(seq)
        cnns = nn.ModuleList(cnns)
        for cnn in cnns:
            cnn[0].weight.data.uniform_(-0.05, 0.05)
            cnn[0].bias.data.fill_(0.)
        self.cnns = cnns
        self.cnn_mix = cnn_mix
        self.kernals = kernals
        # self.highway = Highway(word_emb_size, 4)
        # self.highway.reset_parameters()

    def forward(self, char_emb):
        """

        :param char_emb: expecting size of (seq_len, batch_size, ..., word_len, emb_size)
        :return:
        """

        char_emb = char_emb.transpose(-1, -2)  # (seq_len, batch_size, ..., char_emb_size, word_len)
        size = char_emb.size()
        word_emb_squeeze = char_emb.view(-1, self.char_emb_size, size[-1])  # squeeze the irrelevant axes

        tmp = [_cnn(word_emb_squeeze) for _cnn in self.cnns]
        if self.cnn_mix == 'cat':
            ret = torch.cat(tmp, dim=1)
        elif self.cnn_mix == 'add':
            ret = sum(tmp)
        else:
            raise NotImplementedError

        # recover the squeezed dimensions
        new_size = list(size[:-1])
        new_size[-1] = self.word_emb_size
        assert ret.size(-1) == new_size[-1]
        new_size = tuple(new_size)
        ret = ret.view(new_size)  # (batch, ..., word_emb_size)

        ret = torch.nn.functional.relu(ret)
        # ret = self.highway(ret)

        del char_emb, tmp
        return ret


class RNNComposer(SpellingComposer):

    def __init__(self, char_emb_size, word_emb_size, layers=2):
        super().__init__(char_emb_size, word_emb_size)
        self.rnn = torch.nn.RNN(char_emb_size, word_emb_size // (layers * 2),
                                num_layers=layers,
                                bidirectional=True,
                                batch_first=True)
        self.h0 = torch.nn.Parameter(torch.zeros(word_emb_size // (layers * 2)))

    def forward(self, char_emb, lengths=None):
        """

        :param char_emb: (seq_len, batch_size, ..., word_len, word_emb_dim)
        :param lengths:
        :return: (batch, ..., word_emb_dim)
        """

        input_size = char_emb.size()
        word_len = input_size[-2]
        char_emb_dim = input_size[-1]
        word_emb_dim = len(self.h0) * (self.rnn.num_layers * 2)
        char_emb = char_emb.contiguous().view(-1, word_len, char_emb_dim)  # (seq_len * batch_size ..., word_len, word_emb_dim)
        eq_batch_size = char_emb.size(0)
        h0 = self.h0.unsqueeze(0).unsqueeze(1).expand(self.rnn.num_layers * 2, eq_batch_size, -1).contiguous()
        if lengths is None:
            _, hn = self.rnn(char_emb, h0)
        else:
            packed = torch.nn.utils.rnn.pack_padded_sequence(char_emb, lengths, batch_first=True)
            _, hn = self.rnn(packed, h0)
        hn = hn.transpose(0, 1)  # (seq_len * batch_size * ..., num_layers * num_dir, hidden_size)
        hn = hn.contiguous().view(-1, word_emb_dim)
        ret_size = list(input_size)[0:-2]
        ret_size.append(word_emb_dim)  # (seq_len, batch_size, ..., word_emb_dim)
        hn = hn.view(*tuple(ret_size))

        return hn


class SpellingEmbedding(nn.Embedding):

    def __init__(self, num_embeddings, composer,
                padding_idx=None, max_norm=None, norm_type=2,
                scale_grad_by_freq=False, sparse=False, _weight=None):
        """

        :param num_embeddings:
        :param composer: the module used to perform composition of characters,
            should extend the SpellingComposer class
        :param padding_idx:
        :param max_norm:
        :param norm_type:
        :param scale_grad_by_freq:
        :param sparse:
        """
        super().__init__(num_embeddings, composer.char_emb_size, padding_idx,
                        max_norm, norm_type, scale_grad_by_freq, sparse, _weight)
        nn.init.normal_(self.weight, 0, 0.1)
        if padding_idx is not None:
            nn.init.constant_(self.weight[padding_idx], 0)
        self.composer = composer


    def forward(self, input):
        """
        retrieve the embedding as usual, then apply composition function over the spelling

        :param input: expected to be (batch, ..., word_len)
        :return:
        """
        ret = F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return self.composer(ret)


if __name__ == "__main__":
    input = torch.LongTensor(64, 30, 15).random_(0, 97)
    comp = CNNComposer(13, 40)
    emb = SpellingEmbedding(97, comp)
    output = emb(input)
    print(output.size())
