# -*- coding: UTF-8 -*-

import math
from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from ..tn_cnn import _TNConvNd
from ..tn_linear import _TNLinear


class BTTConv2D(_TNConvNd):
    def __init__(self, in_shape: Union[list, np.ndarray], out_shape: Union[list, np.ndarray],
                 ranks: Union[list, np.ndarray], block_num: int, kernel_size: Union[int, tuple],
                 stride=1, padding=0, bias=True):
        """Block-Term Tucker Decomposition Convolution.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of channel in
        out_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of channel out
        ranks : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^{m+2}`. The rank of the decomposition
        block_num : int
                The number of blocks
        kernel_size : Union[int, tuple]
                The convolutional kernel size
        stride : int
                The length of stride
        padding : int
                The size of padding
        bias : bool
                use bias of convolution or not. ``True`` to use, and ``False`` to not use
        """
        self.block_num = block_num
        super(BTTConv2D, self).__init__(in_shape=in_shape, out_shape=out_shape, ranks=ranks, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias)

        self.reset_parameters()

    def set_tn_type(self):
        """Set as Block-Term Tucker decomposition type.
        """
        self.tn_info["type"] = "btt"

    def set_nodes(self):
        """Generate Block-Term Tucker nodes, then add node information to self.tn_info.
        """
        self.c_factor_num = len(self.ranks) - 2
        self.in_num = len(self.in_shape)
        self.out_num = len(self.out_shape)

        self.core_size = np.prod(self.ranks)

        assert self.in_num == self.out_num, "Factors of input and output is not equal!"
        assert self.in_num == self.c_factor_num, "Number of ranks is not right!"

        blocks_info = []
        for i in range(self.block_num):
            core_info = dict(
                name="core_block%d" % i,
                shape=tuple(self.ranks.tolist())
            )
            core_param_tmp = nn.Parameter(torch.Tensor(*core_info["shape"]))
            setattr(self, core_info["name"], core_param_tmp)

            c_factors_info = []
            for j in range(self.c_factor_num):
                factor_info = dict(
                    name="c_factor%d_block%d" % (j, i),
                    shape=(self.in_shape[j], self.out_shape[j], self.ranks[j])
                )
                factor_param_tmp = nn.Parameter(torch.Tensor(*factor_info["shape"]))
                setattr(self, factor_info["name"], factor_param_tmp)
                c_factors_info.append(factor_info)

            k_factors_info = []
            k_factor_info = dict(
                name="k0_factor_block%d" % i,
                shape=(self.kernel_size[0], self.ranks[-2])
            )
            factor_param_tmp = nn.Parameter(torch.Tensor(*k_factor_info["shape"]))
            setattr(self, k_factor_info["name"], factor_param_tmp)
            k_factors_info.append(k_factor_info)

            k_factor_info = dict(
                name="k1_factor_block%d" % i,
                shape=(self.kernel_size[1], self.ranks[-1])
            )
            factor_param_tmp = nn.Parameter(torch.Tensor(*k_factor_info["shape"]))
            setattr(self, k_factor_info["name"], factor_param_tmp)
            k_factors_info.append(k_factor_info)

            block_info = dict(
                core=core_info,
                c_factors=c_factors_info,
                k_factors=k_factors_info,
            )
            blocks_info.append(block_info)

        self.tn_info["nodes"] = blocks_info

    def set_params_info(self):
        """Record information of Parameters.
        """
        params_ori = self.in_size * self.out_size * np.prod(self.kernel_size)

        params_btt = self.block_num * (np.prod(self.ranks) + np.sum(self.ranks[:-2] * self.in_shape * self.out_shape)
                                       + self.ranks[-2] * self.kernel_size[0] + self.ranks[-1] * self.kernel_size[1])

        compression_ration = params_ori / params_btt

        self.tn_info["t_params"] = params_btt
        self.tn_info["ori_params"] = params_ori
        self.tn_info["cr"] = compression_ration

        print("compression_ration is: ", compression_ration)

    def reset_parameters(self):
        """Reset parameters.
        """
        stdv = 1.0 / math.sqrt(self.out_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def tn_contract(self, inputs: torch.Tensor) -> torch.Tensor:
        """Block-Term Tucker Decomposition Convolution.

        Parameters
        ----------
        inputs : torch.Tensor
                A tensor :math:`\in \mathbb{R}^{b \\times C \\times H \\times W}`

        Returns
        -------
        torch.Tensor
            A tensor :math:`\in \mathbb{R}^{b \\times C' \\times H' \\times W'}`
        """
        weight_tmp = 0
        for block_info in self.tn_info["nodes"]:
            cal_tmp = getattr(self, block_info["core"]["name"])
            c_factors_info = block_info["c_factors"]
            k_factors_info = block_info["k_factors"]

            for j in range(self.in_num):
                factor = getattr(self, c_factors_info[j]["name"])
                cal_tmp = torch.tensordot(cal_tmp, factor, dims=[[0], [2]])

            factor = getattr(self, k_factors_info[0]["name"])
            cal_tmp = torch.tensordot(cal_tmp, factor, dims=[[0], [1]])
            factor = getattr(self, k_factors_info[1]["name"])
            cal_tmp = torch.tensordot(cal_tmp, factor, dims=[[0], [1]])
            weight_tmp += cal_tmp

        in_indexes = [2*i for i in range(self.in_num)]
        out_indexes = [i+1 for i in in_indexes]
        new_index = out_indexes + in_indexes + [self.in_num*2, self.in_num*2+1]
        weight_tmp = weight_tmp.permute(*new_index)
        weight_tmp = weight_tmp.reshape(self.out_size, self.in_size, self.kernel_size[0], self.kernel_size[1])
        res = F.conv2d(inputs, weight_tmp, self.bias, self.stride, self.padding)

        return res

    def forward(self, inputs: torch.Tensor):
        """Block-Term Tucker convolutional forwarding method.

        Parameters
        ----------
        inputs : torch.Tensor
                tensor :math:`\in \mathbb{R}^{b \\times C \\times H \\times W}`

        Returns
        -------
        torch.Tensor
            tensor :math:`\in \mathbb{R}^{b \\times C' \\times H' \\times W'}`
        """
        res = self.tn_contract(inputs)

        return res

    def recover(self):
        """Todo: Use for rebuilding the original tensor.
        """
        pass


class BTTLinear(_TNLinear):
    def __init__(self, in_shape: Union[list, np.ndarray], out_shape: Union[list, np.ndarray],
                 ranks: Union[list, np.ndarray], block_num: int, bias: bool = True):
        """Block-Term Tucker Decomposition Linear.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of feature in
        out_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of feature out
        ranks : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The rank of the decomposition
        block_num : int
                The number of blocks
        bias : bool
                use bias of convolution or not. ``True`` to use, and ``False`` to not use
        """
        self.block_num = block_num
        super(BTTLinear, self).__init__(in_shape=in_shape, out_shape=out_shape, ranks=ranks, bias=bias)
        self.reset_parameters()

    def set_tn_type(self):
        """Set as Block-Term Tucker decomposition type.
        """
        self.tn_info["type"] = "btt"

    def set_nodes(self):
        """Generate Block-Term Tucker nodes, then add node information to self.tn_info.
        """
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
        """Record information of Parameters.
        """
        params_ori = self.in_size * self.out_size
        params_btt = self.block_num * (np.prod(self.ranks) + np.sum(self.ranks * self.in_shape * self.out_shape))
        compression_ration = params_ori / params_btt

        self.tn_info["t_params"] = params_btt
        self.tn_info["ori_params"] = params_ori
        self.tn_info["cr"] = compression_ration

        print("compression_ration is: ", compression_ration)

    def reset_parameters(self):
        """Reset parameters.
        """
        stdv = 1.0 / math.sqrt(self.out_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def tn_contract(self, inputs: torch.Tensor) -> torch.Tensor:
        """Block-Term Tucker linear forwarding method.

        Parameters
        ----------
        inputs : torch.Tensor
                tensor :math:`\in \mathbb{R}^{b \\times C}`

        Returns
        -------
        torch.Tensor
            tensor :math:`\in \mathbb{R}^{b \\times C'}`
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
