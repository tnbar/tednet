# -*- coding: UTF-8 -*-

import math
from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from ..tn_cnn import _TNConvNd
from ..tn_linear import _TNLinear


class TTConv2D(_TNConvNd):
    def __init__(self, in_shape: Union[list, np.ndarray], out_shape: Union[list, np.ndarray],
                 ranks: Union[list, np.ndarray], kernel_size: Union[int, tuple], stride=1, padding=0, bias=True):
        """Tensor Train Decomposition Convolution.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of channel in
        out_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of channel out
        ranks : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The rank of the decomposition
        kernel_size : Union[int, tuple]
                The convolutional kernel size
        stride : int
                The length of stride
        padding : int
                The size of padding
        bias : bool
                use bias of convolution or not. ``True`` to use, and ``False`` to not use
        """
        super(TTConv2D, self).__init__(in_shape=in_shape, out_shape=out_shape, ranks=ranks, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias)
        self.reset_parameters()

    def set_tn_type(self):
        """Set as Tensor Train decomposition type.
        """
        self.tn_info["type"] = "tt"

    def set_nodes(self):
        """Generate Tensor Train nodes, then add node information to self.tn_info.
        """
        self.in_num = len(self.in_shape)
        self.out_num = len(self.out_shape)
        self.core_num = self.in_num

        assert self.in_num == self.out_num == len(self.ranks), "Input and output number should be equal to rank number."

        nodes_info = []
        for i in range(0, self.core_num):
            if i < self.core_num - 1:
                node_info = dict(
                    name="node%d" % i,
                    shape=(self.in_shape[i], self.out_shape[i], self.ranks[i], self.ranks[i + 1])
                )
            else:
                node_info = dict(
                    name="node%d" % i,
                    shape=(self.in_shape[i], self.out_shape[i], self.ranks[i])
                )

            tmp = nn.Parameter(torch.Tensor(*node_info["shape"]))
            self.register_parameter(node_info["name"], tmp)

            nodes_info.append(node_info)

        self.kernel = nn.Conv2d(1, self.ranks[0], self.kernel_size, self.stride, self.padding, bias=False)

        self.tn_info["nodes"] = nodes_info

    def set_params_info(self):
        """Record information of Parameters.
        """
        params_ori = self.in_size * self.out_size * np.prod(self.kernel_size)

        tt_ranks_1 = np.append(self.ranks, 1)
        params_tt = np.sum(tt_ranks_1[:self.in_num] * self.in_shape * self.out_shape * tt_ranks_1[1:(self.in_num + 1)])
        param_kernel = np.prod(self.kernel_size) * self.ranks[0]
        params_tt = params_tt + param_kernel

        compression_ration = params_ori / params_tt

        self.tn_info["t_params"] = params_tt
        self.tn_info["ori_params"] = params_ori
        self.tn_info["cr"] = compression_ration

        print("compression_ration is: ", compression_ration)

    def reset_parameters(self):
        """Reset parameters.
        """
        node_vars = []
        for i in range(self.core_num):
            node_vars.append(1. / (self.in_shape[i] * self.ranks[i]))

        conv_node_var = 2. / (self.kernel_size[0] * self.kernel_size[1])
        std = math.pow(math.sqrt(np.prod(node_vars) * conv_node_var), 1. / (self.core_num + 1))

        for i in range(self.core_num):
            nn.init.normal_(getattr(self, "node%d" % i), std=std)

        nn.init.normal_(self.kernel.weight.data, std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def tn_contract(self, inputs: torch.Tensor)->torch.Tensor:
        """Tensor Train Decomposition Convolution.

        Parameters
        ----------
        inputs : torch.Tensor
                A tensor :math:`\in \mathbb{R}^{b \\times C \\times H \\times W}`

        Returns
        -------
        torch.Tensor
            A tensor :math:`\in \mathbb{R}^{b \\times C' \\times H' \\times W'}`
        """
        batch_size = inputs.shape[0]
        image_hw = inputs.shape[-2:]
        res = inputs.view(-1, 1, *image_hw)

        res = self.kernel(res)
        new_hw = res.shape[-2:]

        res = res.reshape(batch_size, *self.in_shape, self.ranks[0], -1)

        weight_tmp = getattr(self, "node0")
        res = torch.tensordot(res, weight_tmp, dims=([1, -2], [0, 2]))
        for i in range(1, self.core_num):
            weight_tmp = getattr(self, "node%d" % i)
            res = torch.tensordot(res, weight_tmp, dims=([1, -1], [0, 2]))

        res = res.reshape(batch_size, *new_hw, -1)
        return res

    def recover(self):
        """
        Todo: Use for rebuilding the original tensor.
        """
        pass


class TTLinear(_TNLinear):
    def __init__(self, in_shape: Union[list, np.ndarray], out_shape: Union[list, np.ndarray],
                 ranks: Union[list, np.ndarray], bias: bool = True):
        """Tensor Train Decomposition Linear.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of feature in
        out_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of feature out
        ranks : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^{m-1}`. The rank of the decomposition
        bias : bool
                use bias of convolution or not. ``True`` to use, and ``False`` to not use
        """
        super(TTLinear, self).__init__(in_shape=in_shape, out_shape=out_shape, ranks=ranks, bias=bias)

        self.reset_parameters()

    def set_tn_type(self):
        """Set as Tensor Train decomposition type.
        """
        self.tn_info["type"] = "tt"

    def set_nodes(self):
        """Generate Tensor Train nodes, then add node information to self.tn_info.
        """
        self.in_num = len(self.in_shape)
        self.out_num = len(self.out_shape)
        self.nodes_num = self.in_num

        assert self.in_num == self.out_num == len(self.ranks) + 1, \
            "Input and output number should be equal to rank number added 1."

        nodes_info = []
        for i in range(self.nodes_num):
            if i == 0:
                node_info = dict(
                    name="node%d" % i,
                    shape=(self.in_shape[i], self.out_shape[i], self.ranks[i])
                )
                #     0           1          2
                #        0   0        1   1
                #     0          1           2
            elif i < self.nodes_num - 1:
                node_info = dict(
                    name="node%d" % i,
                    shape=(self.in_shape[i], self.out_shape[i], self.ranks[i - 1], self.ranks[i])
                )
            else:
                node_info = dict(
                    name="node%d" % i,
                    shape=(self.in_shape[i], self.out_shape[i], self.ranks[i - 1])
                )

            tmp = nn.Parameter(torch.Tensor(*node_info["shape"]))
            self.register_parameter(node_info["name"], tmp)

            nodes_info.append(node_info)

        self.tn_info["nodes"] = nodes_info

    def set_params_info(self):
        """Record information of Parameters.
        """
        params_ori = self.in_size * self.out_size

        tt_ranks_1 = np.concatenate(([1], self.ranks, [1]))
        params_tt = np.sum(tt_ranks_1[:self.in_num] * self.in_shape * self.out_shape * tt_ranks_1[1:(self.in_num + 1)])

        compression_ration = params_ori / params_tt

        self.tn_info["t_params"] = params_tt
        self.tn_info["ori_params"] = params_ori
        self.tn_info["cr"] = compression_ration

        print("compression_ration is: ", compression_ration)

    def reset_parameters(self):
        """Reset parameters.
        """
        node_vars = []
        node_vars.append(1. / self.in_shape[0])
        for i in range(1, self.nodes_num):
            node_vars.append(1. / (self.in_shape[i] * self.ranks[i - 1]))

        std = math.pow(math.sqrt(np.prod(node_vars)), 1. / self.nodes_num)

        for i in range(self.nodes_num):
            nn.init.normal_(self._parameters["node%d" % i], std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def tn_contract(self, inputs: torch.Tensor)->torch.Tensor:
        """Tensor Train linear forwarding method.

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
        res = inputs.view(-1, *self.in_shape)

        weight_tmp = getattr(self, "node0")
        res = torch.tensordot(res, weight_tmp, dims=([1], [0]))

        for i in range(1, self.nodes_num):
            weight_tmp = getattr(self, "node%d" % i)
            res = torch.tensordot(res, weight_tmp, dims=([1, -1], [0, 2]))

        res = res.reshape(batch_size, -1)
        return res

    def recover(self):
        """
        Todo: Use for rebuilding the original tensor.
        """
        pass



