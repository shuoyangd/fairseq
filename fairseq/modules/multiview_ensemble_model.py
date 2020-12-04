# -*- coding: utf-8 -*-
#
# Copyright © 2020 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2020-12-01
#
# Distributed under terms of the MIT license.

import copy
from ..sequence_generator import  EnsembleModel
from typing import Dict, List
from torch import Tensor

class MultiviewEnsembleModel(EnsembleModel):
    """
    A mock-up wrapper of the original EnsembleModel class for multiview ensemble.
    """

    def __init__(self, model, n_ensemble_views=2):
        """

        :param model: a single model checkpoint used for multi-view ensemble
        :param n_ensemble_views: currently support 2 or 3. 2 -> src & sys; 3 -> src, sys, ref. See paper for details.
        """
        self.n_ensemble_views = n_ensemble_views
        models = [model] * n_ensemble_views  # this will be set to self in parent constructor
        super().__init__(self, models)

    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None

        # in net_input, the token fields we expect are "src_tokens", "sys_tokens" and an optional "ref_tokens" field
        # correspondingly, there will also be length fields "src_lengths", "sys_lengths", "ref_lengths"
        # and there will be other fields that we don't care
        # what we'll do is that we'll make new copies of net_input with only one of them each
        # and pass them to the same model, so we get different ``views'' of the same model
        if "source" in net_input:
            raise NotImplementedError("not sure what format is this, " +
                                      "but we assume the original source sentence is under \"src_tokens\" field")

        # create copies of net_input and remove uncessary fields
        # the model will only concern with the content in "src_tokens"
        # so we also need to move the field to fool the model
        net_input_src = net_input
        net_input_sys = copy.deepcopy(net_input)
        net_input_sys["src_tokens"] = net_input_sys["sys_tokens"]
        net_input_sys["src_lengths"] = net_input_sys["sys_lengths"]
        del net_input_sys["sys_tokens"]
        del net_input_sys["sys_lengths"]
        if self.n_ensemble_views == 3 and "ref_tokens" in net_input:
            net_input_ref = copy.deepcopy(net_input)
            net_input_ref["src_tokens"] = net_input_ref["ref_tokens"]
            net_input_ref["src_lengths"] = net_input_ref["ref_lengths"]
            del net_input_ref["sys_tokens"]
            del net_input_ref["ref_tokens"]
            del net_input_sys["ref_tokens"]
            del net_input_ref["sys_lengths"]
            del net_input_ref["ref_lengths"]
            del net_input_sys["ref_lengths"]
        elif self.n_ensemble_views == 3 and "ref_tokens" in net_input:
            raise ValueError("If you want to do 3-ensemble, you need to pass reference in the input")
        del net_input_src["sys_tokens"]
        del net_input_src["ref_tokens"]
        del net_input_src["sys_lengths"]
        del net_input_src["ref_lengths"]

        net_inputs = [net_input_src, net_input_sys]
        if self.n_ensemble_views == 3 and "ref_tokens" in net_input:
            net_inputs.append(net_input_ref)

        return [model.encoder.forward_torchscript(net_input) for net_input, model in zip(net_inputs, self.models)]
