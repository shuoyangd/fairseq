import torch
import torch.nn as nn

from torch.nn import functional as F


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

    def forward(self, word_emb):
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

    def forward(self, word_emb):
        """

        :param word_emb: expecting size of (batch_size, ..., sent_len, word_len, emb_size)
        :return:
        """

        word_emb = word_emb.transpose(-1, -2)  # (batch, ..., seq_len, char_emb_size)
        size = word_emb.size()  # (batch, ..., char_emb_size, seq_len)
        word_emb_squeeze = word_emb.view(-1, self.char_emb_size, size[-1])  # squeeze the irrelevant axes

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

        del word_emb, tmp
        return ret


class RNNComposer(SpellingComposer):

    def __init__(self, char_emb_size, word_emb_size):
        super().__init__(char_emb_size, word_emb_size)
        raise NotImplementedError

    def forward(self, word_emb):
        raise NotImplementedError


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
        self.composer = composer


    def forward(self, input):
        """
        retrieve the embedding as usual, then apply composition function over the spelling

        :param input: expected to be (batch, ..., sent_len, word_len)
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
