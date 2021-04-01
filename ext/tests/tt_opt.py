# -*- coding: UTF-8 -*-

import tednet

import torch

a = torch.Tensor(6, 8)
b = tednet.to_numpy(a)
c = tednet.to_tensor(b)
d = c.cuda()
e = tednet.to_numpy(d)
f = tednet.to_tensor(e)

