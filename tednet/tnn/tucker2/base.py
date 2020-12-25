# -*- coding: UTF-8 -*-


import math
from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from ..tn_cnn import _TNConvNd
from ..tn_linear import _TNLinear


class TK2Conv2D(_TNConvNd):
    def __init__(self, c_in: int, c_out: int, ranks: Union[list, np.ndarray], kernel_size: Union[int, tuple], stride=1,
                 padding=0, bias=True):
        """Tucker-2 Decomposition Convolution.

        Parameters
        ----------
        c_in : int
                The decomposition shape of channel in
        c_out : int
                The decomposition shape of channel out
        ranks : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^r`. The ranks of the decomposition
        kernel_size : Union[int, tuple]
                1-D param :math:`\in \mathbb{R}^m`. The convolutional kernel size
        stride : int
                The length of stride
        padding : int
                The size of padding
        bias : bool
                use bias of convolution or not. ``True`` to use, and ``False`` to not use
        """
        super(TK2Conv2D, self).__init__(in_shape=[c_in], out_shape=[c_out], ranks=ranks, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias)

        self.reset_parameters()

    def set_tn_type(self):
        """Set as Tucker-2 decomposition type.
        """
        self.tn_info["type"] = "tk2"

    def set_nodes(self):
        """Generate Tucker-2 nodes, then add node information to self.tn_info.
        """
        self.channel_in = self.in_shape[0]
        self.channel_out = self.out_shape[0]

        assert len(self.ranks) == 2, "Length of ranks is %d, not equal 2." % len(self.ranks)

        nodes_info = []

        node_info = dict(
            name="node0",
            shape=(self.channel_in, self.ranks[0])
        )

        tmp = nn.Parameter(torch.Tensor(node_info["shape"][0], node_info["shape"][1]))
        self.register_parameter(node_info["name"], tmp)

        nodes_info.append(node_info)

        node_info = dict(
            name="node1",
            shape=(self.ranks[1], self.channel_out)
        )
        tmp = nn.Parameter(torch.Tensor(node_info["shape"][0], node_info["shape"][1]))
        self.register_parameter(node_info["name"], tmp)

        nodes_info.append(node_info)

        self.kernel = nn.Conv2d(self.ranks[0], self.ranks[1], self.kernel_size, self.stride, self.padding, bias=False)

        self.tn_info["nodes"] = nodes_info

    def set_params_info(self):
        """Record information of Parameters.
        """
        params_ori = self.channel_in * self.channel_out * np.prod(self.kernel_size)

        params_tk2 = self.channel_in * self.ranks[0] + np.prod(self.ranks) * np.prod(self.kernel_size) \
                       + self.ranks[1] * self.channel_out

        compression_ration = params_ori / params_tk2

        self.tn_info["t_params"] = params_tk2
        self.tn_info["ori_params"] = params_ori
        self.tn_info["cr"] = compression_ration

        print("compression_ration is: ", compression_ration)

    def reset_parameters(self):
        """Reset parameters.
        """
        node_vars = []
        node_vars.append(1. / self.channel_in)
        node_vars.append(1. / self.ranks[1])
        conv_node_var = 2. / (self.kernel_size[0] * self.kernel_size[1] * self.ranks[0])
        std = math.pow(math.sqrt(np.prod(node_vars) * conv_node_var), 1. / 3)

        nn.init.normal_(getattr(self, "node0"), std=std)
        nn.init.normal_(getattr(self, "node1"), std=std)
        nn.init.normal_(self.kernel.weight.data, std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def tn_contract(self, inputs: torch.Tensor) -> torch.Tensor:
        """Tucker-2 Decomposition Convolution.

        Parameters
        ----------
        inputs : torch.Tensor
                A tensor :math:`\in \mathbb{R}^{b \\times C \\times H \\times W}`

        Returns
        -------
        torch.Tensor
            A tensor :math:`\in \mathbb{R}^{b \\times C' \\times H' \\times W'}`
        """
        res = inputs

        weight_tmp = getattr(self, "node0")
        res = torch.tensordot(res, weight_tmp, dims=([1], [0]))
        res = res.permute(0, 3, 1, 2)
        res = self.kernel(res)
        weight_tmp = getattr(self, "node1")
        res = torch.tensordot(res, weight_tmp, dims=([1], [0]))

        return res

    def recover(self):
        """
        Todo: Use for rebuilding the original tensor.
        """
        pass


class TK2Linear(_TNLinear):
    def __init__(self, in_shape: Union[list, np.ndarray], out_size: int, ranks: Union[list, np.ndarray],
                 bias: bool = True):
        """Tucker-2 Decomposition Linear.

        +-------------------+--------------------+
        | input length      | ranks length       |
        +===================+====================+
        | 1                 | 1                  |
        +-------------------+--------------------+
        | 3                 | 2                  |
        +-------------------+--------------------+

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m, m \in \{1, 3\}`. The decomposition shape of feature in
        out_size : int
                The output size of the model
        ranks : Union[list, numpy.ndarray]
                1-D param. The rank of the decomposition
        bias : bool
                use bias of convolution or not. ``True`` to use, and ``False`` to not use
        """
        super(TK2Linear, self).__init__(in_shape=in_shape, out_shape=[out_size], ranks=ranks, bias=bias)

        self.reset_parameters()

    def set_tn_type(self):
        """Set as Tucker-2 decomposition type.
        """
        self.tn_info["type"] = "tk2"

    def set_nodes(self):
        """Generate Tucker-2 nodes, then add node information to self.tn_info.
        """
        self.in_num = len(self.in_shape)

        if len(self.ranks) == 2:
            assert self.in_num == 3, "The length of input_shape should be 3."
        elif len(self.ranks) == 1:
            assert self.in_num == 1, "The length of input_shape should be 1."
        else:
            raise ValueError("The length of ranks only has two values: 1 or 2.")

        nodes_info = []

        node_info = dict(
            name="node0",
            shape=(self.in_shape[0], self.ranks[0])
        )
        tmp = nn.Parameter(torch.Tensor(*node_info["shape"]))
        self.register_parameter(node_info["name"], tmp)
        nodes_info.append(node_info)

        if self.in_num > 1:
            node_info = dict(
                name="node_core",
                shape=(self.in_shape[1], self.in_shape[2], self.ranks[0], self.ranks[1])
            )
            tmp = nn.Parameter(torch.Tensor(*node_info["shape"]))
            self.register_parameter(node_info["name"], tmp)
            nodes_info.append(node_info)

            last_node_in = self.ranks[1]
        else:
            last_node_in = self.ranks[0]

        node_info = dict(
            name="node1",
            shape=(last_node_in, self.out_size)
        )
        tmp = nn.Parameter(torch.Tensor(*node_info["shape"]))
        self.register_parameter(node_info["name"], tmp)
        nodes_info.append(node_info)

        self.tn_info["nodes"] = nodes_info

    def set_params_info(self):
        """Record information of Parameters.
        """
        params_ori = np.prod(self.in_shape) * self.out_size

        if self.in_num > 1:
            params_tk2 = self.in_shape[0] * self.ranks[0] \
                           + (self.ranks[0] * self.in_shape[1] * self.in_shape[2] * self.ranks[1]) \
                           + self.ranks[1] * self.out_size
        else:
            params_tk2 = self.in_shape[0] * self.ranks[0] + self.ranks[0] * self.out_size

        compression_ration = params_ori / params_tk2

        self.tn_info["t_params"] = params_tk2
        self.tn_info["ori_params"] = params_ori
        self.tn_info["cr"] = compression_ration

        print("compression_ration is: ", compression_ration)

    def reset_parameters(self):
        """Reset parameters.
        """
        node_vars = []
        node_vars.append(1. / self.in_shape[0])
        if self.in_num > 1:
            node_vars.append(1. / (self.ranks[0] * self.in_shape[1] * self.in_shape[2]))
            last_node_in = self.ranks[1]
        else:
            last_node_in = self.ranks[0]
        node_vars.append(1. / last_node_in)

        std = math.pow(math.sqrt(np.prod(node_vars)), 1. / len(node_vars))

        nn.init.normal_(getattr(self, "node0"), std=std)
        nn.init.normal_(getattr(self, "node1"), std=std)
        if self.in_num > 1:
            nn.init.normal_(getattr(self, "node_core"), std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def tn_contract(self, inputs: torch.Tensor)->torch.Tensor:
        """Tucker-2 linear forwarding method.

        Parameters
        ----------
        inputs : torch.Tensor
                tensor :math:`\in \mathbb{R}^{b \\times C}`

        Returns
        -------
        torch.Tensor
            tensor :math:`\in \mathbb{R}^{b \\times C'}`
        """
        if self.in_num > 1:
            res = inputs.view(-1, *self.in_shape)
            weight_tmp = getattr(self, "node0")
            res = torch.tensordot(res, weight_tmp, dims=([1], [0]))
            weight_tmp = getattr(self, "node_core")
            res = torch.tensordot(res, weight_tmp, dims=([1, 2, 3], [0, 1, 2]))
        else:
            res = inputs
            weight_tmp = getattr(self, "node0")
            res = torch.tensordot(res, weight_tmp, dims=([1], [0]))

        weight_tmp = getattr(self, "node1")
        res = torch.tensordot(res, weight_tmp, dims=([1], [0]))

        return res

    def recover(self):
        """
        Todo: Use for rebuilding the original tensor.
        """
        pass



