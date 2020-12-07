# -*- coding: UTF-8 -*-

import sys
# sys.path.append('../tednet')

import torch
from tednet.tnn.tensor_ring import TRLeNet5


class Test_TR(object):
    def test_bmnist(self):
        model = TRLeNet5(10, rs=[6, 6, 6, 6])

        data = torch.Tensor(16, 1, 28, 28)

        res = model(data)

        assert len(res.shape) == 2
