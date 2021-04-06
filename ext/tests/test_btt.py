# -*- coding: UTF-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath('../tednet'))

import tednet as tdt
import tednet.tnn.bt_tucker as btt

import torch


class Test_BTT:
    def test_BTTConv2D(self):
        print('\n BTTConv2D')
        data = torch.Tensor(16, 1, 28, 28)
        model = btt.BTTConv2D([1, 1], [4, 5], [6, 6, 6, 6], 3, 2)
        res = model(data)

        assert len(res.shape) == 4

    def test_BTTLinear(self):
        print('\n BTTLinear')
        data = torch.Tensor(16, 20)
        model = btt.BTTLinear([4, 5], [2, 5], [6, 6], 2)
        res = model(data)

        assert len(res.shape) == 2

    def test_BTTLeNet5(self):
        print('\n BTTLeNet5')
        data = torch.Tensor(16, 1, 28, 28)
        model = btt.BTTLeNet5(10, rs=[6, 6, 6, 6])
        res = model(data)

        assert len(res.shape) == 2

    def test_BTTResNet20(self):
        print('\n BTTResNet20')
        data = torch.Tensor(16, 3, 28, 28)
        model = btt.BTTResNet20([6, 6, 6, 6, 6, 6, 6], 10)
        res = model(data)

        assert len(res.shape) == 2

    def test_BTTResNet32(self):
        print('\n BTTResNet32')
        data = torch.Tensor(16, 3, 28, 28)
        model = btt.BTTResNet32([6, 6, 6, 6, 6, 6, 6], 10)
        res = model(data)

        assert len(res.shape) == 2

    def test_BTTLSTM(self):
        print('\n BTTLSTM')
        from collections import namedtuple
        LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

        data = torch.Tensor(10, 16, 6)
        model = btt.BTTLSTM([2, 3], [3, 3], [6, 6], 2)

        state = LSTMState(torch.zeros(16, 9),
                          torch.zeros(16, 9))
        res, _ = model(data, state)

        assert len(res.shape) == 3


