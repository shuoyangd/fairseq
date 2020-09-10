# -*- coding: utf-8 -*-
#
# Copyright © 2020 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2020-08-24
#
# Distributed under terms of the MIT license.

import pdb
import sys
import torch

sas = torch.load(sys.argv[1])
for sa in sas:
  sa = sa[0]
  sa = torch.clamp(sa, min=0.0)
  z = torch.sum(sa, dim=1)
  sa = (sa.transpose(0, 1) / z).transpose(0, 1)
  print(sa.tolist())
