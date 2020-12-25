# -*- coding: UTF-8 -*-

import math
from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from ..tn_cnn import _TNConvNd
from ..tn_linear import _TNLinear


class TRConv2D(_TNConvNd):
    def __init__(self, in_shape: Union[list, np.ndarray], out_shape: Union[list, np.ndarray],
                 ranks: Union[list, np.ndarray], kernel_size: Union[int, tuple], stride=1, padding=0, bias=True):
        """Tensor Ring Decomposition Convolution.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of channel in
        out_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^n`. The decomposition shape of channel out
        ranks : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^{m+n+1}`. The ranks of the decomposition
        kernel_size : Union[int, tuple]
                The convolutional kernel size
        stride : int
                The length of stride
        padding : int
                The size of padding
        bias : bool
                use bias of convolution or not. ``True`` to use, and ``False`` to not use
        """
        super(TRConv2D, self).__init__(in_shape=in_shape, out_shape=out_shape, ranks=ranks, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias)

        self.reset_parameters()

    def set_tn_type(self):
        """Set as Tensor Ring decomposition type.
        """
        self.tn_info["type"] = "tr"

    def set_nodes(self):
        """Generate Tensor Ring nodes, then add node information to self.tn_info.
        """
        self.in_num = len(self.in_shape)
        self.out_num = len(self.out_shape)

        self.nodes_num = self.in_num + self.out_num
        self.ranks_fill = np.append(self.ranks, self.ranks[0])

        assert self.nodes_num + 1 == len(self.ranks), "The number of ranks is not suitable."

        nodes_info = []
        for i in range(self.nodes_num):
            if i < self.in_num:
                left_rank = self.ranks_fill[i]
                right_rank = self.ranks_fill[i + 1]
                middle_rank = self.in_shape[i]

                node_info = dict(
                    name="in_node%d" % i,
                    shape=(left_rank, middle_rank, right_rank)
                )
            else:
                out_i = i - self.in_num

                left_rank = self.ranks_fill[i + 1]
                right_rank = self.ranks_fill[i + 2]
                middle_rank = self.out_shape[out_i]

                node_info = dict(
                    name="out_node%d" % out_i,
                    shape=(left_rank, middle_rank, right_rank)
                )

            node = nn.Parameter(torch.Tensor(*node_info["shape"]))
            self.register_parameter(node_info["name"], node)

            nodes_info.append(node_info)

        self.kernel_cin = self.ranks_fill[self.in_num]
        self.kernel_cout = self.ranks_fill[self.in_num + 1]
        self.kernel = nn.Conv2d(self.kernel_cin, self.kernel_cout, self.kernel_size, self.stride,
                                self.padding, bias=False)

        self.tn_info["nodes"] = nodes_info

    def set_params_info(self):
        """Record information of Parameters.
        """
        params_ori = self.in_size * self.out_size * np.prod(self.kernel_size)

        params_in = np.sum(self.ranks_fill[:self.in_num] * self.in_shape * self.ranks_fill[1:(self.in_num+1)])
        params_kernel = np.prod(self.kernel_size) * self.kernel_cin * self.kernel_cout
        params_out = np.sum(self.ranks_fill[(self.in_num+1):-1] * self.out_shape * self.ranks_fill[(self.in_num+2):])
        params_tr = params_in + params_kernel + params_out

        compression_ration = params_ori / params_tr

        self.tn_info["t_params"] = params_tr
        self.tn_info["ori_params"] = params_ori
        self.tn_info["cr"] = compression_ration

        print("compression_ration is: ", compression_ration)

    def reset_parameters(self):
        """Reset parameters.
        """
        node_vars = []
        for i in range(self.in_num):
            if i == 0:
                node_vars.append(1./self.in_size)
            else:
                node_vars.append(1./self.ranks_fill[i])
        for i in range(self.out_num):
            if i == 0:
                node_vars.append(1./(self.ranks_fill[0]*self.ranks_fill[self.in_num + 1]))
            else:
                node_vars.append(1./self.ranks_fill[self.in_num + 1])

        conv_node_var = 2./(self.kernel_size[0]*self.kernel_size[1]*self.ranks_fill[self.in_num])
        std = math.pow(math.sqrt(np.prod(node_vars)*conv_node_var), 1./(self.nodes_num + 1))

        for i in range(self.in_num):
            nn.init.normal_(getattr(self, "in_node%d" % i), std=std)
        for i in range(self.out_num):
            nn.init.normal_(getattr(self, "out_node%d" % i), std=std)

        nn.init.normal_(self.kernel.weight.data, std=std)

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
            A tensor :math:`\in \mathbb{R}^{b \\times H' \\times W' \\times C'}`
        """
        batch_size = inputs.shape[0]
        image_hw = inputs.shape[2:]

        # res: [b, I0, I1, I2, H, W]
        res = inputs.view(batch_size, *self.in_shape, *image_hw)

        I_in = getattr(self, "in_node0")
        for i in range(1, self.in_num):
            weight_tmp = getattr(self, "in_node%d" % i)
            I_in = torch.tensordot(I_in, weight_tmp, dims=([-1], [0]))

        # res: [b, H, W, r0, r3]
        in_positions = list(range(1, self.in_num+1))
        res = torch.tensordot(res, I_in, dims=(in_positions, in_positions))

        # res: [b, H, W, r0, r3] | res: [br0, r3, H, W]
        res = res.permute(0, 3, 4, 1, 2).contiguous()
        res = res.reshape(-1, self.ranks_fill[self.in_num], *image_hw)

        # res: [bHW, r0r3] | res: [br0, r4, nH, nW]
        res = self.kernel(res)
        image_new_hw = res.shape[2:]

        # res: [b, r0, r4, nHnW]
        res = res.reshape(batch_size, self.ranks_fill[0], self.ranks_fill[self.in_num + 1], -1)

        O_out = getattr(self, "out_node0")
        for i in range(1, self.out_num):
            weight_tmp = getattr(self, "out_node%d" % i)
            O_out = torch.tensordot(O_out, weight_tmp, dims=([-1], [0]))

        # res: [b, nH, nW, O0O1O2]
        res = torch.tensordot(res, O_out, dims=([1, 2], [-1, 0]))
        res = res.reshape(batch_size, *image_new_hw, self.out_size)

        return res

    def recover(self):
        """Todo: Use for rebuilding the original tensor.
        """
        pass


class TRLinear(_TNLinear):
    def __init__(self, in_shape: Union[list, np.ndarray], out_shape: Union[list, np.ndarray],
                 ranks: Union[list, np.ndarray], bias: bool = True):
        """The Tensor Ring Decomposition Linear.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^m`. The decomposition shape of feature in
        out_shape : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^n`. The decomposition shape of feature out
        ranks : Union[list, numpy.ndarray]
                1-D param :math:`\in \mathbb{R}^{m+n}`. The ranks of linear
        bias : bool
                use bias of linear or not. ``True`` to use, and ``False`` to not use
        """
        super(TRLinear, self).__init__(in_shape=in_shape, out_shape=out_shape, ranks=ranks, bias=bias)

        self.reset_parameters()

    def set_tn_type(self):
        """Set as Tensor Ring decomposition type.
        """
        self.tn_info["type"] = "tr"

    def set_nodes(self):
        """Generate tensor ring nodes, then add node information to self.tn_info.
        """
        self.in_num = len(self.in_shape)
        self.out_num = len(self.out_shape)

        self.nodes_num = self.in_num + self.out_num
        self.nodes_shape = np.append(self.in_shape, self.out_shape)
        self.ranks_fill = np.append(self.ranks, self.ranks[0])

        assert self.nodes_num == len(self.ranks), "The number of ranks is not suitable."

        nodes_info = []
        for i in range(self.nodes_num):
            left_rank = self.ranks_fill[i]
            right_rank = self.ranks_fill[i + 1]
            middle_rank = self.nodes_shape[i]

            node_info = dict(
                name="node%d" % i,
                shape=(left_rank, middle_rank, right_rank)
            )
            tmp = nn.Parameter(torch.Tensor(*node_info["shape"]))
            self.register_parameter(node_info["name"], tmp)

            nodes_info.append(node_info)

        self.tn_info["nodes"] = nodes_info

    def set_params_info(self):
        """Record information of Parameters.
        """
        params_ori = self.in_size * self.out_size
        params_tr = np.sum(np.multiply(np.multiply(self.ranks_fill[:-1], self.ranks_fill[1:]), self.nodes_shape))
        compression_ration = params_ori / params_tr

        self.tn_info["t_params"] = params_tr
        self.tn_info["ori_params"] = params_ori
        self.tn_info["cr"] = compression_ration

        print("compression_ration is: ", compression_ration)

    def reset_parameters(self):
        """Reset parameters.
        """
        node_vars = []
        for i in range(self.in_num):
            if i == 0:
                node_vars.append(1. / self.in_size)
            else:
                node_vars.append(1. / self.ranks_fill[i])
        for i in range(self.out_num):
            if i == 0:
                node_vars.append(1. / (self.ranks_fill[0] * self.ranks_fill[self.in_num + 1]))
            else:
                node_vars.append(1. / self.ranks_fill[self.in_num + i])
        std = math.pow(math.sqrt(np.prod(node_vars)), 1. / self.nodes_num)

        for i in range(self.nodes_num):
            nn.init.normal_(getattr(self, "node%d" % i), std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def tn_contract(self, inputs: torch.Tensor)->torch.Tensor:
        """Tensor Ring linear forwarding method.

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

        # res: [b, I0, I1, I2] | res: [b, r0, r3]
        I0 = getattr(self, "node0")
        res = torch.tensordot(res, I0, dims=([1], [1]))
        for i in range(1, self.in_num):
            weight_tmp = getattr(self, "node%d" % i)
            res = torch.tensordot(res, weight_tmp, dims=([1, -1], [1, 0]))

        for i in range(0, self.out_num-1):
            weight_tmp = getattr(self, "node%d" % (self.in_num + i))
            res = torch.tensordot(res, weight_tmp, dims=([-1], [0]))
        O_L = getattr(self, "node%d" % (self.in_num + self.out_num-1))
        res = torch.tensordot(res, O_L, dims=([1, -1], [2, 0])).reshape(batch_size, -1)

        return res

    def recover(self):
        """Todo: Use for rebuilding the original tensor.
        """
        pass
