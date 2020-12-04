# -*- coding: UTF-8 -*-

from .base import TRConv2D, TRLinear
from .tr_lenet import TRLeNet5
from .tr_resnet import TRResNet18, TRResNet34
from .tr_rnn import TRLSTM

__all__ = ["TRConv2D", "TRLinear",
           "TRLeNet5",
           "TRResNet18", "TRResNet34",
           "TRLSTM"
           ]
