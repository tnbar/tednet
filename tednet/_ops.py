# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from collections import namedtuple


def hard_sigmoid(tensor: torch.Tensor):
    """Computes element-wise hard sigmoid of x.
    See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279

    Parameters
    ----------
    tensor : torch.Tensor
             tensor :math:`\in \mathbb{R}^{{i_1} \\times \dots \\times {i_n}}`

    Returns
    -------
    torch.Tensor
        tensor :math:`\in \mathbb{R}^{{i_1} \\times \dots \\times {i_n}}`
    """
    tensor = (0.2 * tensor) + 0.5
    tensor = F.threshold(-tensor, -1, -1)
    tensor = F.threshold(-tensor, 0, 0)
    return tensor


def eye(n: int, m: int, device: torch.device = "cpu", requires_grad: bool=False):
    """Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : int
             the number of rows
    m : int
             the number of columns
    device : torch.device
             the desired device of returned tensor. Default will be the ``CPU``.
    requires_grad : bool
             If autograd should record operations on the returned tensor. Default: ``False``.

    Returns
    -------
    torch.Tensor
        2-D tensor :math:`\in \mathbb{R}^{{i_1} \\times {i_2}}`
    """
    return torch.eye(n=n, m=m, device=device, requires_grad=requires_grad)


def ones(x):
    # TODO:
    pass


def prod_all(x):
    # TODO:
    pass


def diag(x):
    # TODO:
    pass


def tensordot(x):
    # TODO:
    pass


def from_numpy(x):
    # TODO:
    pass
