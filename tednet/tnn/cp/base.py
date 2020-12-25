# -*- coding: UTF-8 -*-

import math
from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from ..tn_cnn import _TNConvNd
from ..tn_linear import _TNLinear


class CPConv2D(_TNConvNd):
    def __init__(self, c_in: int, c_out: int, rank: int, kernel_size: Union[int, tuple],
                 stride=1, padding=0, bias=True):
        """CANDECOMP/PARAFAC Decomposition Convolution.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of channel in
        out_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^n`. The decomposition shape of channel out
        rank : int
                The rank of the decomposition
        kernel_size : Union[int, tuple]
                The convolutional kernel size
        stride : int
                The length of stride
        padding : int
                The size of padding
        bias : bool
                use bias of convolution or not. ``True`` to use, and ``False`` to not use
        """
        super(CPConv2D, self).__init__(in_shape=[c_in], out_shape=[c_out], ranks=[rank], kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias)

        self.reset_parameters()

    def set_tn_type(self):
        """Set as CANDECOMP/PARAFAC decomposition type.
        """
        self.tn_info["type"] = "cp"

    def set_nodes(self):
        """Generate CANDECOMP/PARAFAC nodes, then add node information to self.tn_info.
        """
        self.channel_in = self.in_shape[0]
        self.channel_out = self.out_shape[0]

        assert len(self.ranks) == 1, "Length of ranks is %d, not equal 1." % len(self.ranks)

        blocks_info = []
        for i in range(self.ranks[0]):
            block_info = []
            node_info = dict(
                name="node_in_block%d" % i,
                shape=self.channel_in
            )

            tmp = nn.Parameter(torch.Tensor(node_info["shape"]))
            self.register_parameter(node_info["name"], tmp)

            block_info.append(node_info)

            node_info = dict(
                name="node_out_block%d" % i,
                shape=self.channel_out
            )

            tmp = nn.Parameter(torch.Tensor(node_info["shape"]))
            self.register_parameter(node_info["name"], tmp)

            block_info.append(node_info)

            node_info = dict(
                name="node_k0_block%d" % i,
                shape=self.kernel_size[0]
            )

            tmp = nn.Parameter(torch.Tensor(node_info["shape"]))
            self.register_parameter(node_info["name"], tmp)

            block_info.append(node_info)

            node_info = dict(
                name="node_k1_block%d" % i,
                shape=self.kernel_size[1]
            )

            tmp = nn.Parameter(torch.Tensor(node_info["shape"]))
            self.register_parameter(node_info["name"], tmp)

            block_info.append(node_info)

            blocks_info.append(block_info)

        self.tn_info["nodes"] = blocks_info

    def set_params_info(self):
        """Record information of Parameters.
        """
        params_ori = self.in_size * self.out_size * np.prod(self.kernel_size)

        params_cp = (self.channel_in + self.channel_out + self.kernel_size[0] + self.kernel_size[1]) * self.ranks[0]

        compression_ration = params_ori / params_cp

        self.tn_info["t_params"] = params_cp
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

    def tn_contract(self, inputs: torch.Tensor)->torch.Tensor:
        """Tensor Decomposition Convolution.

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
        for i in range(self.ranks[0]):
            weight_out = getattr(self, "node_out_block%d" % i)
            weight_out = weight_out.unsqueeze(-1)

            weight_in = getattr(self, "node_in_block%d" % i)
            weight_in = weight_in.unsqueeze(-1)

            weight_out_in = torch.tensordot(weight_out, weight_in, dims=[[-1], [-1]]).unsqueeze(-1)

            weight_k0 = getattr(self, "node_k0_block%d" % i)
            weight_k0 = weight_k0.unsqueeze(-1)

            weight_k1 = getattr(self, "node_k1_block%d" % i)
            weight_k1 = weight_k1.unsqueeze(-1)

            weight_k0_k1 = torch.tensordot(weight_k0, weight_k1, dims=[[-1], [-1]]).unsqueeze(-1)

            weight_tmp += torch.tensordot(weight_out_in, weight_k0_k1, dims=[[-1], [-1]])

        res = F.conv2d(inputs, weight_tmp, self.bias, self.stride, self.padding)

        return res

    def forward(self, inputs: torch.Tensor):
        """Tensor convolutional forwarding method.

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


class CPLinear(_TNLinear):
    def __init__(self, in_shape: Union[list, np.ndarray], out_shape: Union[list, np.ndarray],
                 rank: int, bias: bool = True):
        """The CANDECOMP/PARAFAC Decomposition Linear.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of feature in
        out_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^n`. The decomposition shape of feature out
        ranks : int
                The rank of linear
        bias : bool
                use bias of linear or not. ``True`` to use, and ``False`` to not use
        """
        super(CPLinear, self).__init__(in_shape=in_shape, out_shape=out_shape, ranks=[rank], bias=bias)

        self.reset_parameters()

    def set_tn_type(self):
        """Set as CANDECOMP/PARAFAC decomposition type.
        """
        self.tn_info["type"] = "cp"

    def set_nodes(self):
        """Generate tensor ring nodes, then add node information to self.tn_info.
        """
        self.in_num = len(self.in_shape)
        self.out_num = len(self.out_shape)

        self.nodes_num = self.in_num + self.out_num
        self.nodes_shape = np.append(self.in_shape, self.out_shape)

        blocks_info = []
        for i in range(self.ranks[0]):
            block_info = []
            for j in range(self.nodes_num):
                node_info = dict(
                    name="node%d_block%d" % (j, i),
                    shape=self.nodes_shape[j]
                )
                tmp = nn.Parameter(torch.Tensor(node_info["shape"]))
                self.register_parameter(node_info["name"], tmp)

                block_info.append(node_info)

            blocks_info.append(block_info)

        self.tn_info["nodes"] = blocks_info

    def set_params_info(self):
        """Record information of Parameters.
        """
        params_ori = self.in_size * self.out_size
        params_cp = np.sum(self.nodes_shape) * self.ranks[0]
        compression_ration = params_ori / params_cp

        self.tn_info["t_params"] = params_cp
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

    def tn_contract(self, inputs: torch.Tensor)->torch.Tensor:
        """CANDECOMP/PARAFAC linear forwarding method.

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
        inputs_dec = inputs.view(-1, *self.in_shape)

        res = 0
        for i in range(self.ranks[0]):
            weight_tmp = getattr(self, "node%d_block%d" % (0, i))
            res_tmp = torch.tensordot(inputs_dec, weight_tmp, dims=[[1], [0]])
            for j in range(1, self.in_num):
                weight_tmp = getattr(self, "node%d_block%d" % (j, i))
                res_tmp = torch.tensordot(res_tmp, weight_tmp, dims=[[1], [0]])
            for j in range(self.in_num, self.nodes_num):
                weight_tmp = getattr(self, "node%d_block%d" % (j, i))
                res_tmp = torch.tensordot(res_tmp.unsqueeze(-1), weight_tmp.unsqueeze(-1), dims=[[-1], [-1]])

            res += res_tmp

        res = res.reshape(batch_size, -1)

        return res

    def recover(self):
        """Todo: Use for rebuilding the original tensor.
        """
        pass
