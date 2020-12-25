# -*- coding: UTF-8 -*-

import abc
from typing import Union

import torch
import torch.nn as nn

import numpy as np

__all__ = ["_TNBase"]


class _TNBase(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, in_shape: Union[list, np.ndarray], out_shape: Union[list, np.ndarray],
                 ranks: Union[list, np.ndarray], bias: bool = True):
        """The basis of tensor decomposition networks.

        Parameters
        ----------
        in_shape : Union[list, numpy.ndarray]
                 1-D param :math:`\in \mathbb{R}^m`
        out_shape : Union[list, numpy.ndarray]
                 1-D param :math:`\in \mathbb{R}^n`
        ranks : Union[list, numpy.ndarray]
                 1-D param :math:`\in \mathbb{R}^r`
        bias : bool
                 use bias or not. ``True`` to use, and ``False`` to not use
        """
        super(_TNBase, self).__init__()
        if isinstance(in_shape, np.ndarray):
            self.in_shape = in_shape
        else:
            self.in_shape = np.array(in_shape)

        if isinstance(out_shape, np.ndarray):
            self.out_shape = out_shape
        else:
            self.out_shape = np.array(out_shape)

        if isinstance(ranks, np.ndarray):
            self.ranks = ranks
        else:
            self.ranks = np.array(ranks)

        self.check_setting()

        self.tn_info = dict()
        self.in_size = np.prod(in_shape)
        self.out_size = np.prod(out_shape)

        self.set_tn_type()
        self.set_nodes()
        self.set_params_info()

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

    def check_setting(self):
        """Check whether in_shape, out_shape, ranks are 1-D params.
        """
        assert len(self.in_shape.shape) == 1, "The in_shape must be 1-d array."
        assert len(self.out_shape.shape) == 1, "The out_shape must be 1-d array."
        assert len(self.ranks.shape) == 1, "The ranks must be 1-d array."

    @abc.abstractmethod
    def set_tn_type(self):
        """Set the tensor decomposition type.
        The types are as follows:

        +------------+------------------------+
        | type       | tensor decomposition   |
        +============+========================+
        | tr         | Tensor Ring            |
        +------------+------------------------+
        | tt         | Tensor Train           |
        +------------+------------------------+
        | tk2        | Tucker2                |
        +------------+------------------------+
        | cp         | CANDECAMP/PARAFAC      |
        +------------+------------------------+
        | btt        | Block-Term Tucker      |
        +------------+------------------------+

        Examples:
            >>> tn_type = "tr"
            >>> self.tn_info["type"] = tn_type
        """
        pass


    @abc.abstractmethod
    def set_nodes(self):
        """Generate tensor nodes, then add node information to self.tn_info.

        Examples:
            >>> nodes_info = []
            >>> node_info = dict(name="node1", shape=[2, 3, 4])
            >>> nodes_info.append(node_info)
            >>> self.tn_info["nodes"] = nodes_info
        """
        pass

    @abc.abstractmethod
    def set_params_info(self):
        """Record information of Parameters.

        Examples:
            >>> self.tn_info["t_params"] = tn_parameters
            >>> self.tn_info["ori_params"] = ori_parameters
            >>> self.tn_info["cr"] = ori_parameters / tn_parameters
        """
        pass

    @abc.abstractmethod
    def tn_contract(self, inputs: torch.Tensor) -> torch.Tensor:
        """The method of contract inputs and tensor nodes.

        Parameters
        ----------
        inputs : torch.Tensor
                 tensor :math:`\in \mathbb{R}^{{i_1} \\times \dots \\times {i_m}}`

        Returns
        -------
        torch.Tensor
            tensor :math:`\in \mathbb{R}^{{i_1} \\times \dots \\times {i_n}}`
        """
        pass

    @abc.abstractmethod
    def recover(self):
        """Use for rebuilding the original tensor.
        """
        pass


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        """Tensor Ring Block.

        Parameters
        ----------
        lambd :
                a lambda function.
        """
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forwarding method.

        Parameters
        ----------
        inputs : torch.Tensor
                tensor :math:`\in \mathbb{R}^{b \\times C \\times H \\times W}`

        Returns
        -------
        torch.Tensor
            tensor :math:`\in \mathbb{R}^{b \\times C' \\times H' \\times W'}`
        """
        return self.lambd(inputs)
