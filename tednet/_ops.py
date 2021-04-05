# -*- coding: UTF-8 -*-

from collections import namedtuple

import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


def hard_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
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


def eye(n: int, m: int, device: torch.device = "cpu", requires_grad: bool=False) -> torch.Tensor:
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
        2-D tensor :math:`\in \mathbb{R}^{{n} \\times {m}}`
    """
    return torch.eye(n=n, m=m, device=device, requires_grad=requires_grad)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch.Tensor to numpy.ndarray.

    Parameters
    ----------
    tensor : torch.Tensor
             tensor :math:`\in \mathbb{R}^{{i_1} \\times \dots \\times {i_n}}`

    Returns
    -------
    numpy.ndarray
        arr :math:`\in \mathbb{R}^{{i_1} \\times \dots \\times {i_n}}`
    """
    if tensor.device.type == "cpu":
        arr = tensor.numpy()
    else:
        arr = tensor.cpu().numpy()

    return arr


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy.ndarray to torch.Tensor.

    Parameters
    ----------
    arr : numpy.ndarray
             arr :math:`\in \mathbb{R}^{{i_1} \\times \dots \\times {i_n}}`

    Returns
    -------
    torch.Tensor
        tensor :math:`\in \mathbb{R}^{{i_1} \\times \dots \\times {i_n}}`
    """
    tensor = torch.from_numpy(arr)

    return tensor
