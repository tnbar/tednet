# -*- coding: UTF-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath('../tednet'))

import tednet as tdt
import torch
import numpy as np


class Test_Ops:
    def test_eye(self):
        print('\n eye function')
        tensor = tdt.eye(6, 6)
        assert isinstance(tensor, torch.Tensor)

    def test_hard_sigmoid(self):
        print('\n hard_sigmoid function')
        tensor = tdt.eye(6, 6)
        tensor = tdt.hard_sigmoid(tensor)
        assert isinstance(tensor, torch.Tensor)

    def test_to_numpy(self):
        print('\n to_numpy function')
        tensor = tdt.eye(6, 6)
        assert isinstance(tensor, torch.Tensor)
        tensor = tdt.to_numpy(tensor)
        assert isinstance(tensor, np.ndarray)

    def test_to_tensor(self):
        print('\n to_tensor function')
        tensor = np.eye(6, 6)
        assert isinstance(tensor, np.ndarray)
        tensor = tdt.to_tensor(tensor)
        assert isinstance(tensor, torch.Tensor)
