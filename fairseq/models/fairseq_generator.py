# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.nn as nn
import torch.nn.functional as F


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
