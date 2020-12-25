# -*- coding: UTF-8 -*-

import os
from config import proj_cfg

import tednet.tnn.tensor_ring as tr
import tednet.tnn.bt_tucker as btt
import tednet.tnn.cp as cp
import tednet.tnn.tensor_train as tt
import tednet.tnn.tucker2 as tk2


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
    dataset_path = os.path.join(proj_cfg.root_path, './datasets/features')

    tr_rnn_ = tr.TRLinear([64, 32], [32*4, 64], [40, 60, 48, 48])
    num_para_calcular(tr_rnn_)

