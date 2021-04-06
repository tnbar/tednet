# -*- coding: UTF-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath('../tednet'))

import tednet as tdt
import tednet.tnn.tucker2 as tk2

import torch


class Test_TK2:
    def test_TK2Conv2D(self):
        print('\n TK2Conv2D')
        data = torch.Tensor(16, 1, 28, 28)
        model = tk2.TK2Conv2D(1, 20, [6, 6], 3)
        res = model(data)

        assert len(res.shape) == 4

    def test_TK2Linear(self):
        print('\n TK2Linear')
        data = torch.Tensor(16, 20)
        model = tk2.TK2Linear([2, 2, 5], 10, [6, 6])
        res = model(data)

        assert len(res.shape) == 2

    def test_TK2LeNet5(self):
        print('\n TK2LeNet5')
        data = torch.Tensor(16, 1, 28, 28)
        model = tk2.TK2LeNet5(10, rs=[6, 6, 6, 6])
        res = model(data)

        assert len(res.shape) == 2

    def test_TK2ResNet20(self):
        print('\n TK2ResNet20')
        data = torch.Tensor(16, 3, 28, 28)
        model = tk2.TK2ResNet20([6, 6, 6, 6, 6, 6, 6], 10)
        res = model(data)

        assert len(res.shape) == 2

    def test_TK2ResNet32(self):
        print('\n TK2ResNet32')
        data = torch.Tensor(16, 3, 28, 28)
        model = tk2.TK2ResNet32([6, 6, 6, 6, 6, 6, 6], 10)
        res = model(data)

        assert len(res.shape) == 2

    def test_TK2LSTM(self):
        print('\n TK2LSTM')
        from collections import namedtuple
        LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

        data = torch.Tensor(10, 16, 6)
        model = tk2.TK2LSTM([6], 9, [6])

        state = LSTMState(torch.zeros(16, 9),
                          torch.zeros(16, 9))
        res, _ = model(data, state)

        assert len(res.shape) == 3


