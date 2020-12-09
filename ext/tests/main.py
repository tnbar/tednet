# -*- coding: UTF-8 -*-

import torch

import tednet.tnn as tnn
from tednet.tnn import tucker2, cp


# input_c = torch.Tensor(4, 6, 28, 28)
# input_l = torch.Tensor(4, 6)
# tl = tensor_ring.TRLinear([2, 3], [4, 5, 6], [7, 8, 9, 10, 11])
# t = tensor_ring.TRLenet5(10)

# t = bt_tucker.BTTLinear([2, 3], [4, 5], [6, 7], 3)

# tc = tensor_train.TTConv2D([2, 3], [4, 5], [6, 7], 3)
# tl = tensor_train.TTLinear([2, 3], [4, 5], [6])

# input_c = torch.Tensor(4, 6, 28, 28)
# tc = tensor_train.TTConv2D([2, 3], [4, 5], [6, 7], 3)
# input_l1 = torch.Tensor(4, 6)
# tl1 = tucker2.TK2Linear([6], 20, [10])
# res = tl1(input_l1)
# input_l2 = torch.Tensor(4, 24)
# tl2 = tucker2.TK2Linear([2, 3, 4], 20, [10, 11])
# res = tl2(input_l2)

# input_l1 = torch.Tensor(4, 1, 28, 28)
# tl1 = tensor_train.TTLenet5(10)
# res = tl1(input_l1)

# input_l1 = torch.Tensor(4, 1, 28, 28)
# tl1 = tucker2.TK2Lenet5(10)
# res = tl1(input_l1)

# input_l1 = torch.Tensor(4, 3, 28, 28)
# tl1 = tensor_ring.TRResNet18([6, 6, 6, 6, 6, 6], 100)
# res = tl1(input_l1)

# input_l1 = torch.Tensor(4, 3, 28, 28)
# tl1 = tensor_train.TTResNet18([6, 6, 6, 6, 6, 6], 100)
# res = tl1(input_l1)

# input_l1 = torch.Tensor(4, 3, 28, 28)
# tl1 = tucker2.TK2ResNet18([6, 6, 6, 6, 6, 6], 100)
# res = tl1(input_l1)

# input_l1 = torch.Tensor(4, 3, 28, 28)
# tl1 = cp.CPConv2D(3, 16, 3, 3)
# res = tl1(input_l1)
input_l2 = torch.Tensor(4, 24)
tl2 = cp.CPLinear([2, 3, 4], [4, 5], 4)
res = tl2(input_l2)

# x = tdt.eye(4, 3)

pass
