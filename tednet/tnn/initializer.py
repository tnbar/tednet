# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn

__all__ = ["trunc_normal_init", "normal", "uniform"]


def _truncated_normal_(tensor, mean=0., std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


def trunc_normal_init(model, mean: float = 0., std: float = 0.1):
    """Initialize network with truncated normal distribution

    Parameters
    ----------
    model : Any
            a model needed to be initialized
    mean : float
            mean of truncated normal distribution
    std : float
            standard deviation of truncated normal distribution
    """
    for weight in model.parameters():
        _truncated_normal_(weight.data, mean, std)

    if model.bias is not None:
        nn.init.zeros_(model.bias)


def normal(model, mean: float = 0., std: float = 0.1):
    """Initialize network with standard normal distribution

    Parameters
    ----------
    model : Any
            a model needed to be initialized
    mean : float
            mean of normal distribution
    std : float
            standard deviation of normal distribution
    """
    for weight in model.parameters():
        nn.init.normal_(weight.data, mean, std)

    if model.bias is not None:
        nn.init.zeros_(model.bias)


def uniform(model, a: float = 0., b: float = 1.):
    """Initialize network with standard uniform distribution

    Parameters
    ----------
    model : Any
            a model needed to be initialized
    a : float
            the lower bound of the uniform distribution
    b : float
            the upper bound of the uniform distribution
    """
    for weight in model.parameters():
        nn.init.uniform_(weight.data, a, b)

    if model.bias is not None:
        nn.init.zeros_(model.bias)
