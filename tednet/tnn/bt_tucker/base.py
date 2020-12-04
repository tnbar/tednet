# -*- coding: UTF-8 -*-

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from ..tn_module import _TNLinear


class BTTLinear(_TNLinear):
    def __init__(self, in_shape: list, out_shape: list, ranks: list, block_num: int, bias: bool = True):
        self.block_num = block_num
        super(BTTLinear, self).__init__(in_shape=in_shape, out_shape=out_shape, ranks=ranks, bias=bias)
        self.reset_parameters()

    def set_tn_type(self):
        """
        Set as block-term tucker decomposition type.
        """
        self.tn_info["type"] = "btt"

    def set_nodes(self):
        self.factor_num = len(self.ranks)
        self.in_num = len(self.in_shape)
        self.out_num = len(self.out_shape)

        self.core_size = np.prod(self.ranks)

        assert self.in_num == self.out_num == self.factor_num, "Factors is not equal!"

        blocks_info = []
        for i in range(self.block_num):
            core_info = dict(
                name="core_block%d" % i,
                shape=tuple(self.ranks.tolist())
            )
            core_param_tmp = nn.Parameter(torch.Tensor(*core_info["shape"]))
            setattr(self, core_info["name"], core_param_tmp)

            factors_info = []
            for j in range(self.factor_num):
                factor_info = dict(
                    name="factor%d_block%d" % (j, i),
                    shape=(self.in_shape[j], self.out_shape[j], self.ranks[j])
                )
                factor_param_tmp = nn.Parameter(torch.Tensor(*factor_info["shape"]))
                setattr(self, factor_info["name"], factor_param_tmp)
                factors_info.append(factor_info)

            block_info = dict(
                core=core_info,
                factors=factors_info
            )
            blocks_info.append(block_info)

        self.tn_info["nodes"] = blocks_info

    def set_params_info(self):
        params_ori = self.in_size * self.out_size
        params_btt = self.block_num * (np.prod(self.ranks) + np.sum(self.ranks * self.in_shape * self.out_shape))
        compression_ration = params_ori / params_btt

        self.tn_info["t_params"] = params_btt
        self.tn_info["ori_params"] = params_ori
        self.tn_info["cr"] = compression_ration

        print("compression_ration is: ", compression_ration)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def tn_contract(self, inputs: torch.Tensor)->torch.Tensor:
        """
        The method of contract inputs and tensor nodes.
        @param inputs: [b, C]
        @return: [b, C']
        """
        batch_size = inputs.shape[0]
        in_tmp = inputs.view(-1, *self.in_shape.tolist())
        # res: [b, I0, I1, I2] | res: [b, I2, I1, I0]
        trans_tmp = [0]
        trans_tmp.extend([self.in_num - i for i in range(self.in_num)])
        in_tmp = in_tmp.permute(*trans_tmp)

        res = 0
        for block_info in self.tn_info["nodes"]:
            core = getattr(self, block_info["core"]["name"])
            factors_info = block_info["factors"]
            factor_offset = 1

            cal_tmp = in_tmp
            for factor_info in factors_info:
                factor = getattr(self, factor_info["name"])
                I, J, R = factor.shape
                cal_tmp = cal_tmp.reshape(-1, I).matmul(factor.view(I, -1))
                cal_tmp = cal_tmp.view(batch_size*factor_offset, -1, J*R)
                cal_tmp = cal_tmp.permute(0, 2, 1)
                factor_offset *= J

            cal_tmp = cal_tmp.reshape(-1, self.core_size).matmul(core.view(self.core_size, -1))
            cal_tmp = cal_tmp.view(batch_size, -1)
            res += cal_tmp
        return res

    def recover(self):
        """
        Todo: Use for rebuilding the original tensor.
        """
        pass
