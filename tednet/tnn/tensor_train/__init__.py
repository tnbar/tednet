# -*- coding: UTF-8 -*-

from .base import TTConv2D, TTLinear
from .tt_lenet import TTLeNet5
from .tt_resnet import TTResNet20, TTResNet32
from .tt_rnn import TTLSTM

__all__ = ["TTConv2D", "TTLinear",
           "TTLeNet5",
           "TTResNet20", "TTResNet32",
           "TTLSTM"
           ]
