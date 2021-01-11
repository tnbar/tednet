# -*- coding: UTF-8 -*-

import os
from config import proj_cfg

from model.resnet_cifar10 import resnet32_cifar10
from model.resnet_cifar100 import resnet32_cifar100

from model.resnet_tn_cifar10 import *
from model.resnet_tn_cifar100 import *


def num_para_calcular(net, name):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print(r"%s Total Params：%d" %(name, k))


if __name__ == '__main__':
    dataset_path = os.path.join(proj_cfg.root_path, './datasets/')

    # rn32_c10 = resnet32_cifar10(dataset_path)
    # num_para_calcular(rn32_c10, "rn32_c10")
    # rn32_c100 = resnet32_cifar100(dataset_path)
    # num_para_calcular(rn32_c100, "rn32_c10")
    #
    # btt_rn32_c10 = BTTResNet32_CIFAR10([4, 4, 4, 4, 4, 4, 4], dataset_path)
    # num_para_calcular(btt_rn32_c10, "btt_rn32_c10")
    # btt_rn32_c100 = BTTResNet32_CIFAR100([4, 4, 4, 4, 4, 4, 4], dataset_path)
    # num_para_calcular(btt_rn32_c100, "btt_rn32_c100")

    # cp_rn32_c10 = CPResNet32_CIFAR10([10, 10, 10, 10, 10, 10, 10], dataset_path)
    # num_para_calcular(cp_rn32_c10, "cp_rn32_c10")
    # cp_rn32_c100 = CPResNet32_CIFAR100([10, 10, 10, 10, 10, 10, 10], dataset_path)
    # num_para_calcular(cp_rn32_c100, "cp_rn32_c100")

    # tk2_rn32_c10 = TK2ResNet32_CIFAR10([10, 10, 10, 10, 10, 10, 10], dataset_path)
    # num_para_calcular(tk2_rn32_c10, "tk2_rn32_c10")
    # tk2_rn32_c100 = TK2ResNet32_CIFAR100([10, 10, 10, 10, 10, 10, 10], dataset_path)
    # num_para_calcular(tk2_rn32_c100, "tk2_rn32_c100")

    # tr_rn32_c10 = TRResNet32_CIFAR10([10, 10, 10, 10, 10, 10, 10], dataset_path)
    # num_para_calcular(tr_rn32_c10, "tr_rn32_c10")
    # tr_rn32_c100 = TRResNet32_CIFAR100([10, 10, 10, 10, 10, 10, 10], dataset_path)
    # num_para_calcular(tr_rn32_c100, "tr_rn32_c100")

    tt_rn32_c10 = TTResNet32_CIFAR10([10, 10, 10, 10, 10, 10, 10], dataset_path)
    num_para_calcular(tt_rn32_c10, "tt_rn32_c10")
    tt_rn32_c100 = TTResNet32_CIFAR100([10, 10, 10, 10, 10, 10, 10], dataset_path)
    num_para_calcular(tt_rn32_c100, "tt_rn32_c100")

