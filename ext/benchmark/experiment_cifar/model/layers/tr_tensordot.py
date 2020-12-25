# -*- coding: UTF-8 -*-

# Author: Perry
# @Create Time: 2020/5/24 00:28

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


class TensorRingConvolution(torch.nn.Module):
    def __init__(self, input_rank, output_rank, input_shape, output_shape, kernel_size, stride=1, padding=0, bias=True, init="ours_res_relu"):
        super(TensorRingConvolution, self).__init__()

        assert len(input_rank) == len(input_shape) + 1
        assert len(output_rank) == len(output_shape)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.input_rank = np.array(input_rank)
        self.input_shape = np.array(input_shape)
        self.input_size = np.prod(self.input_shape)

        self.output_rank = np.array(output_rank)
        self.output_rank_complement = np.append(output_rank, self.input_rank[0])
        self.output_shape = np.array(output_shape)
        self.output_size = np.prod(self.output_shape)

        self.kernel_channel_in = self.input_rank[-1]
        self.kernel_channel_out = self.output_rank[0]
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.input_num = len(self.input_shape)
        self.output_num = len(self.output_shape)
        self.nodes_num = self.input_num + self.output_num

        # self.weights = []
        node_shapes = self.generate_node_shapes()
        for node_shape in node_shapes:
            tmp = nn.Parameter(torch.Tensor(*node_shape["shape"]))
            # self.weights.append(tmp)
            self.register_parameter(node_shape["name"], tmp)
            # self.weights.append(getattr(self, node_shape["name"]))
        self.kernel = nn.Conv2d(self.kernel_channel_in, self.kernel_channel_out,
                                self.kernel_size, self.stride, self.padding, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size))
        else:
            self.register_parameter('bias', None)

        self.calculate_compression()

        self.init = init
        if self.init == "ours_res_relu":
            self.reset_parameters_ours_res_relu()
        elif self.init == "ours_conv_relu":
            self.reset_parameters_ours_conv_relu()
        elif self.init == "ours_conv_linear":
            self.reset_parameters_ours_conv_linear()
        elif self.init == "ours_std_conv_relu":
            self.reset_parameters_ours_std_conv_relu()
        elif self.init == "ours_std_conv_linear":
            self.reset_parameters_ours_std_conv_linear()
        elif self.init == "ours_std_res_relu":
            self.reset_parameters_ours_std_res_relu()
        elif "normal" in self.init:
            self.reset_parameters_normal(self.init)
        # elif self.init == "kaiming":
        #     self.reset_parameters_kaiming()
        # elif self.init == "xavier":
        #     self.reset_parameters_xavier()
        else:
            raise KeyError("The initialization %s is not existed." % self.init)

    def forward(self, inputs):
        res = self.tensor_contract(inputs)
        if self.bias is not None:
            # fused op is marginally faster
            res = torch.add(self.bias, res)

        # res: [b, H, W, O1O2O3O4] | res: [b, O1O2O3O4, H, W]
        res = res.permute(0, 3, 1, 2)
        res = res.contiguous()

        return res

    def reset_parameters_normal(self, init):
        std = float("0." + init.split("_")[1])
        for weight in self.parameters():
            nn.init.normal_(weight.data, 0, std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters_ours_res_relu(self):
        nn.init.normal_(self._parameters["input_node%d" % 0], std=math.sqrt(1./self.input_size))
        for i in range(1, self.input_num):
            nn.init.kaiming_normal_(self._parameters["input_node%d" % i], mode="fan_in", nonlinearity="linear")

        nn.init.normal_(self._parameters["output_node%d" % 0], std=math.sqrt(math.sqrt(2.)/(self.input_rank[0]*self.output_rank[0])))
        for i in range(1, self.output_num):
            nn.init.kaiming_normal_(self._parameters["output_node%d" % i], mode="fan_in", nonlinearity="linear")

        nn.init.kaiming_normal_(self.kernel.weight.data, mode="fan_in", nonlinearity="linear")

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters_ours_conv_relu(self):
        nn.init.normal_(self._parameters["input_node%d" % 0], std=math.sqrt(1./self.input_size))
        for i in range(1, self.input_num):
            nn.init.kaiming_normal_(self._parameters["input_node%d" % i], mode="fan_in", nonlinearity="linear")

        nn.init.normal_(self._parameters["output_node%d" % 0], std=math.sqrt(1./(self.input_rank[0]*self.output_rank[0])))
        for i in range(1, self.output_num):
            nn.init.kaiming_normal_(self._parameters["output_node%d" % i], mode="fan_in", nonlinearity="linear")

        nn.init.kaiming_normal_(self.kernel.weight.data, mode="fan_in", nonlinearity="relu")

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters_ours_conv_linear(self):
        nn.init.normal_(self._parameters["input_node%d" % 0], std=math.sqrt(1./self.input_size))
        for i in range(1, self.input_num):
            nn.init.kaiming_normal_(self._parameters["input_node%d" % i], mode="fan_in", nonlinearity="linear")

        nn.init.normal_(self._parameters["output_node%d" % 0], std=math.sqrt(1./(self.input_rank[0]*self.output_rank[0])))
        for i in range(1, self.output_num):
            nn.init.kaiming_normal_(self._parameters["output_node%d" % i], mode="fan_in", nonlinearity="linear")

        nn.init.kaiming_normal_(self.kernel.weight.data, mode="fan_in", nonlinearity="linear")

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters_ours_std_conv_linear(self):
        node_vars = []
        for i in range(self.input_num):
            if i == 0:
                node_vars.append(1./self.input_size)
            else:
                node_vars.append(1./self.input_rank[i])
        for i in range(self.output_num):
            if i == 0:
                node_vars.append(1./self.input_rank[0]*self.output_rank[0])
            else:
                node_vars.append(1./self.output_rank[i])

        conv_node_var = 1./(self.kernel_size[0]*self.kernel_size[1]*self.input_rank[-1])
        std = math.pow(math.sqrt(np.prod(node_vars)*conv_node_var), 1./(self.nodes_num+1))

        for i in range(self.input_num):
            nn.init.normal_(self._parameters["input_node%d" % i], std=std)
        for i in range(self.output_num):
            nn.init.normal_(self._parameters["output_node%d" % i], std=std)

        nn.init.normal_(self.kernel.weight.data, std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters_ours_std_conv_relu(self):
        node_vars = []
        for i in range(self.input_num):
            if i == 0:
                node_vars.append(1./self.input_size)
            else:
                node_vars.append(1./self.input_rank[i])
        for i in range(self.output_num):
            if i == 0:
                node_vars.append(1./self.input_rank[0]*self.output_rank[0])
            else:
                node_vars.append(1./self.output_rank[i])

        conv_node_var = 2./(self.kernel_size[0]*self.kernel_size[1]*self.input_rank[-1])
        std = math.pow(math.sqrt(np.prod(node_vars)*conv_node_var), 1./(self.nodes_num+1))

        for i in range(self.input_num):
            nn.init.normal_(self._parameters["input_node%d" % i], std=std)
        for i in range(self.output_num):
            nn.init.normal_(self._parameters["output_node%d" % i], std=std)

        nn.init.normal_(self.kernel.weight.data, std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters_ours_std_res_relu(self):
        node_vars = []
        for i in range(self.input_num):
            if i == 0:
                node_vars.append(1./self.input_size)
            else:
                node_vars.append(1./self.input_rank[i])
        for i in range(self.output_num):
            if i == 0:
                node_vars.append(math.sqrt(2.)/self.input_rank[0]*self.output_rank[0])
            else:
                node_vars.append(1./self.output_rank[i])

        conv_node_var = 1./(self.kernel_size[0]*self.kernel_size[1]*self.input_rank[-1])
        std = math.pow(math.sqrt(np.prod(node_vars)*conv_node_var), 1./(self.nodes_num+1))

        for i in range(self.input_num):
            nn.init.normal_(self._parameters["input_node%d" % i], std=std)
        for i in range(self.output_num):
            nn.init.normal_(self._parameters["output_node%d" % i], std=std)

        nn.init.normal_(self.kernel.weight.data, std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def adjust_parameters_ours_std_res_relu(self):
        node_vars = []
        for i in range(self.input_num):
            if i == 0:
                node_vars.append(1./self.input_size)
            else:
                node_vars.append(1./self.input_rank[i])
        for i in range(self.output_num):
            if i == 0:
                node_vars.append(math.sqrt(2.)/self.input_rank[0]*self.output_rank[0])
            else:
                node_vars.append(1./self.output_rank[i])

        conv_node_var = 1./(self.kernel_size[0]*self.kernel_size[1]*self.input_rank[-1])
        std = math.pow(math.sqrt(np.prod(node_vars)*conv_node_var), 1./(self.nodes_num+1))

        for i in range(self.input_num):
            param = self._parameters["input_node%d" % i]
            ori_std = torch.std(param.data)
            param.data = param.data / ori_std * std
        for i in range(self.output_num):
            param = self._parameters["output_node%d" % i]
            ori_std = torch.std(param.data)
            param.data = param.data / ori_std * std

        # nn.init.normal_(self.kernel.weight.data, std=std)
        param = self.kernel.weight
        ori_std = torch.std(param.data)
        param.data = param.data / ori_std * std

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def generate_node_shapes(self):
        node_shapes = []
        for i in range(self.nodes_num):
            if i < self.input_num:
                left_rank = self.input_rank[i]
                right_rank = self.input_rank[i + 1]
                middle_rank = self.input_shape[i]

                tmp = dict(
                    name="input_node%d" % i,
                    shape=(left_rank, middle_rank, right_rank)
                )
            else:
                output_i = i - self.input_num

                left_rank = self.output_rank_complement[output_i]
                right_rank = self.output_rank_complement[output_i + 1]
                middle_rank = self.output_shape[output_i]

                tmp = dict(
                    name="output_node%d" % output_i,
                    shape=(left_rank, middle_rank, right_rank)
                )
            node_shapes.append(tmp)
        return node_shapes

    def tensor_contract(self, inputs):
        batch_size = inputs.shape[0]
        image_hw = inputs.shape[2:]

        # res: [b, I0, I1, I2, H, W]
        res = inputs.view(batch_size, *self.input_shape, *image_hw)

        I_in = getattr(self, "input_node0")
        for i in range(1, self.input_num):
            weight_tmp = getattr(self, "input_node%d" % i)
            I_in = torch.tensordot(I_in, weight_tmp, dims=([-1], [0]))

        # res: [b, H, W, r0, r3]
        input_positions = list(range(1, self.input_num+1))
        res = torch.tensordot(res, I_in, dims=(input_positions, input_positions))

        # res: [b, H, W, r0, r3] | res: [br0, r3, H, W]
        res = res.permute(0, 3, 4, 1, 2)
        res = res.reshape(-1, self.input_rank[-1], *image_hw)

        #### Dropout

        # res: [bHW, r0r3] | res: [br0, r4, nH, nW]
        res = self.kernel(res)
        image_new_hw = res.shape[2:]

        # res: [b, r0, r4, nHnW]
        res = res.reshape(batch_size, self.input_rank[0], self.output_rank[0], -1)

        ##### Dropout

        O_out = getattr(self, "output_node0")
        for i in range(1, self.output_num):
            weight_tmp = getattr(self, "output_node%d" % i)
            O_out = torch.tensordot(O_out, weight_tmp, dims=([-1], [0]))

        # res: [b, nH, nW, O0O1O2]
        res = torch.tensordot(res, O_out, dims=([1, 2], [-1, 0]))
        res = res.reshape(batch_size, *image_new_hw, self.output_size)

        return res

    def calculate_compression(self):
        param_origin = self.input_size * self.output_size * np.prod(self.kernel_size)

        param_input = np.sum(self.input_rank[:-1] * self.input_shape * self.input_rank[1:])
        param_kernel = np.prod(self.kernel_size) * self.kernel_channel_in * self.kernel_channel_out
        param_output = np.sum(self.output_rank_complement[:-1] * self.output_shape * self.output_rank_complement[1:])
        param_tr = param_input + param_kernel + param_output

        compression_ration = param_origin / param_tr
        print("compression_ration is: ", compression_ration)
        return compression_ration



class TensorRingLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, input_shape: list,
                 output_shape: list, rank_shape: list, bias: bool = True, init="ours_res_relu"):
        super(TensorRingLinear, self).__init__()

        # The size of the original matrix
        self.input_size = input_size
        self.output_size = output_size

        # The shape of the tensor ring decomposition
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.rank_shape = rank_shape

        # Check whether shapes are right
        self.check_shape_setting()

        self.nodes_num = len(self.rank_shape)
        self.input_num = len(self.input_shape)
        self.output_num = len(self.output_shape)

        assert self.input_num + self.output_num == self.nodes_num

        self.tr_ranks_line = self.rank_shape + [self.rank_shape[0]]
        self.whole_node_shape = self.input_shape + self.output_shape

        # self.weights = []
        node_shapes = self.generate_node_shapes()
        for node_shape in node_shapes:
            tmp = nn.Parameter(torch.Tensor(*node_shape["shape"]))
            # self.weights.append(tmp)
            self.register_parameter(node_shape["name"], tmp)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size))
        else:
            self.register_parameter('bias', None)

        self.calculate_compression()

        self.init = init
        if self.init == "ours_linear":
            self.reset_parameters_ours_linear()
        elif self.init == "ours_relu":
            self.reset_parameters_ours_relu()
        elif self.init == "ours_std_linear":
            self.reset_parameters_ours_std_linear()
        elif self.init == "ours_std_relu":
            self.reset_parameters_ours_std_relu()
        elif "normal" in self.init:
            self.reset_parameters_normal(self.init)
        # elif self.init == "kaiming":
        #     self.reset_parameters_kaiming()
        # elif self.init == "xavier":
        #     self.reset_parameters_xavier()
        else:
            raise KeyError("The initialization %s is not existed." % self.init)

    def forward(self, inputs):
        res = self.tensor_contract(inputs)
        if self.bias is not None:
            # fused op is marginally faster
            res = torch.add(self.bias, res)
        return res

    def reset_parameters_normal(self, init):
        std = float("0." + init.split("_")[1])
        for weight in self.parameters():
            nn.init.normal_(weight.data, 0, std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters_ours_linear(self):
        nn.init.normal_(self._parameters["node%d" % 0], std=math.sqrt(1. / self.input_size))
        for i in range(1, self.input_num):
            nn.init.kaiming_normal_(self._parameters["node%d" % i], mode="fan_in", nonlinearity="linear")

        nn.init.normal_(self._parameters["node%d" % self.input_num],
                        std=math.sqrt(1. / (self.tr_ranks_line[self.input_num] * self.tr_ranks_line[-1])))
        for i in range(self.input_num + 1, self.nodes_num):
            nn.init.kaiming_normal_(self._parameters["node%d" % i], mode="fan_in", nonlinearity="linear")

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters_ours_relu(self):
        nn.init.normal_(self._parameters["node%d" % 0], std=math.sqrt(1. / self.input_size))
        for i in range(1, self.input_num):
            nn.init.kaiming_normal_(self._parameters["node%d" % i], mode="fan_in", nonlinearity="linear")

        nn.init.normal_(self._parameters["node%d" % self.input_num],
                        std=math.sqrt(2. / (self.tr_ranks_line[self.input_num] * self.tr_ranks_line[-1])))
        for i in range(self.input_num + 1, self.nodes_num):
            nn.init.kaiming_normal_(self._parameters["node%d" % i], mode="fan_in", nonlinearity="linear")

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters_ours_std_relu(self):
        node_vars = []
        for i in range(self.input_num):
            if i == 0:
                node_vars.append(1. / self.input_size)
            else:
                node_vars.append(1. / self.tr_ranks_line[i])
        for i in range(self.output_num):
            if i == 0:
                node_vars.append(2. / (self.tr_ranks_line[0] * self.tr_ranks_line[self.input_num + 1]))
            else:
                node_vars.append(1. / self.tr_ranks_line[self.input_num + i])
        std = math.pow(math.sqrt(np.prod(node_vars)), 1. / self.nodes_num)

        for i in range(self.nodes_num):
            nn.init.normal_(self._parameters["node%d" % i], std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters_ours_std_linear(self):
        node_vars = []
        for i in range(self.input_num):
            if i == 0:
                node_vars.append(1. / self.input_size)
            else:
                node_vars.append(1. / self.tr_ranks_line[i])
        for i in range(self.output_num):
            if i == 0:
                node_vars.append(1. / (self.tr_ranks_line[0] * self.tr_ranks_line[self.input_num + 1]))
            else:
                node_vars.append(1. / self.tr_ranks_line[self.input_num + i])
        std = math.pow(math.sqrt(np.prod(node_vars)), 1. / self.nodes_num)

        for i in range(self.nodes_num):
            nn.init.normal_(self._parameters["node%d" % i], std=std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def adjust_parameters_ours_std_linear(self):
        node_vars = []
        for i in range(self.input_num):
            if i == 0:
                node_vars.append(1. / self.input_size)
            else:
                node_vars.append(1. / self.tr_ranks_line[i])
        for i in range(self.output_num):
            if i == 0:
                node_vars.append(1. / (self.tr_ranks_line[0] * self.tr_ranks_line[self.input_num + 1]))
            else:
                node_vars.append(1. / self.tr_ranks_line[self.input_num + i])
        std = math.pow(math.sqrt(np.prod(node_vars)), 1. / self.nodes_num)

        for i in range(self.nodes_num):
            param = self._parameters["node%d" % i]
            ori_std = torch.std(param.data)
            param.data = param.data / ori_std * std

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # To avoid forgetting to design shapes wrongly
    def check_shape_setting(self):
        assert self.input_size == np.prod(self.input_shape), "The decomposition of the input_size is not suitable!"
        assert self.output_size == np.prod(
            self.output_shape), "The decomposition of the output_size is not suitable!"
        # print("The Tensor Ring shape is qualified!")

    def generate_node_shapes(self):
        node_shapes = []
        for i in range(self.nodes_num):
            left_rank = self.tr_ranks_line[i]
            right_rank = self.tr_ranks_line[i + 1]
            middle_rank = self.whole_node_shape[i]

            tmp = dict(
                name="node%d" % i,
                shape=(left_rank, middle_rank, right_rank)
            )
            node_shapes.append(tmp)
        return node_shapes

    # def tensor_contract(self, inputs):
    #     batch_size = inputs.shape[0]
    #     res = inputs
    #
    #     I_in = getattr(self, "node0")
    #     for i in range(1, self.input_num):
    #         # weight_tmp = self.weights[i]
    #         weight_tmp = getattr(self, "node%d" % i)
    #         I_in = torch.tensordot(I_in, weight_tmp, dims=([-1], [0]))
    #
    #     # I_in: [r0I0, I1r2] | I_in: [r0r2, I0I1]
    #     I_in = I_in.reshape(self.tr_ranks_line[0], -1, self.tr_ranks_line[self.input_num])
    #     I_in = I_in.permute(1, 0, 2)
    #     I_in = I_in.reshape(self.input_size, -1)
    #
    #     # res: [b, I0, I1, I2], I_in: [r0r2, I0I1] | res: [b, r0r2]
    #     res = torch.matmul(res, I_in)
    #
    #     # O_out = self.weights[self.input_num]
    #     O_out = getattr(self, "node%d" % self.input_num)
    #     for i in range(1, self.output_num):
    #         # weight_tmp = self.weights[self.input_num + i]
    #         weight_tmp = getattr(self, "node%d" % (self.input_num + i))
    #         O_out = torch.tensordot(O_out, weight_tmp, dims=([-1], [0]))
    #
    #     # O_out: [r2O0, O1r0] | I_in: [I0I1, r0r2]
    #     O_out = O_out.reshape(self.tr_ranks_line[self.input_num], -1, self.tr_ranks_line[-1])
    #     O_out = O_out.permute(2, 0, 1)
    #     O_out = O_out.reshape(self.tr_ranks_line[self.input_num]*self.tr_ranks_line[-1], -1)
    #
    #     # res: [b, r0r2], I_in: [I0I1, r0r2] | res: [b, I0I1]
    #     res = torch.matmul(res, O_out).reshape(batch_size, -1)
    #
    #     return res

    def tensor_contract(self, inputs):
        batch_size = inputs.shape[0]
        res = inputs.view(-1, *self.input_shape)

        # res: [b, I0, I1, I2] | res: [b, r0, r3]
        I0 = getattr(self, "node0")
        res = torch.tensordot(res, I0, dims=([1], [1]))
        for i in range(1, self.input_num):
            # weight_tmp = self.weights[i]
            weight_tmp = getattr(self, "node%d" % i)
            res = torch.tensordot(res, weight_tmp, dims=([1, -1], [1, 0]))

        for i in range(0, self.output_num-1):
            # weight_tmp = self.weights[self.input_num + i]
            weight_tmp = getattr(self, "node%d" % (self.input_num + i))
            res = torch.tensordot(res, weight_tmp, dims=([-1], [0]))
        O_L = getattr(self, "node%d" % (self.input_num + self.output_num-1))
        res = torch.tensordot(res, O_L, dims=([1, -1], [2, 0])).reshape(batch_size, -1)

        return res

    def calculate_compression(self):
        param_origin = self.input_size * self.output_size
        param_tr = np.sum(np.multiply(np.multiply(self.tr_ranks_line[:-1], self.tr_ranks_line[1:]), self.whole_node_shape))
        compression_ration = param_origin / param_tr
        print("compression_ration is: ", compression_ration)
        return compression_ration
