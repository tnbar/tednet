# -*- coding: UTF-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath('../tednet'))

import tednet as tdt
import tednet.tnn.tensor_train as tt

import torch


class Test_TT:
    def test_TTConv2D(self):
        print('\n TTConv2D')
        data = torch.Tensor(16, 1, 28, 28)
        model = tt.TTConv2D([1, 1], [4, 5], [6, 6], 3)
        res = model(data)

        assert len(res.shape) == 4

    def test_TTLinear(self):
        print('\n TTLinear')
        data = torch.Tensor(16, 20)
        model = tt.TTLinear([4, 5], [2, 5], [6])
        res = model(data)

        assert len(res.shape) == 2

    def test_TTLeNet5(self):
        print('\n TTLeNet5')
        data = torch.Tensor(16, 1, 28, 28)
        model = tt.TTLeNet5(10, rs=[6, 6, 6, 6])
        res = model(data)

        assert len(res.shape) == 2

    def test_TTResNet20(self):
        print('\n TTResNet20')
        data = torch.Tensor(16, 3, 28, 28)
        model = tt.TTResNet20([6, 6, 6, 6, 6, 6, 6], 10)
        res = model(data)

        assert len(res.shape) == 2

    def test_TTResNet32(self):
        print('\n TTResNet32')
        data = torch.Tensor(16, 3, 28, 28)
        model = tt.TTResNet32([6, 6, 6, 6, 6, 6, 6], 10)
        res = model(data)

        assert len(res.shape) == 2

    def test_TTLSTM(self):
        print('\n TTLSTM')
        from collections import namedtuple
        LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

        data = torch.Tensor(10, 16, 6)
        model = tt.TTLSTM([2, 3], [3, 3], [6])

        state = LSTMState(torch.zeros(16, 9),
                          torch.zeros(16, 9))
        res, _ = model(data, state)

        assert len(res.shape) == 3


