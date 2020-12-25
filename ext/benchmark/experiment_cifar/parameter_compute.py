# -*- coding: UTF-8 -*-

import os
from config import proj_cfg

from model.resnet_tn_cifar10 import TRResNet20_CIFAR10, TRResNet32_CIFAR10

from model.resnet_cifar100 import resnet20_cifar100, resnet32_cifar100
from model.resnet_tn_cifar100 import TRResNet20_CIFAR100, TRResNet32_CIFAR100


def num_para_calcular(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
    #     print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
    #     print("该层参数和：" + str(l))
        k = k + l
    print(r"Total Params：" + str(k))


if __name__ == '__main__':
    dataset_path = os.path.join(proj_cfg.root_path, './datasets/')

    tr_res32_c10 = TRResNet32_CIFAR10([2, 2, 2, 2, 2, 2, 2, 2], dataset_path=dataset_path)
    num_para_calcular(tr_res32_c10)

    tr_res32_c100 = TRResNet32_CIFAR100([6, 6, 6, 6, 6, 6, 6, 6], dataset_path=dataset_path)
    num_para_calcular(tr_res32_c100)

