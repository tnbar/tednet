# -*- coding: UTF-8 -*-

import os
from config import proj_cfg

import tednet.tnn.tensor_ring as tr
import tednet.tnn.bt_tucker as btt
import tednet.tnn.cp as cp
import tednet.tnn.tensor_train as tt
import tednet.tnn.tucker2 as tk2


if __name__ == '__main__':
    dataset_path = os.path.join(proj_cfg.root_path, './datasets/features')

    tr_rnn_ = tr.TRLSTM([64, 32], [32, 64], [20, 20, 20, 20])
    btt_rnn_ = btt.BTTLSTM([64, 32], [32, 64], [5, 5], 2)
    cp_rnn_ = cp.CPLSTM([64, 32], [32, 64], 400)
    tt_rnn_ = tt.TTLSTM([64, 32], [32, 64], [10])
    tk2_rnn_ = tk2.TK2LSTM([16, 16, 8], 2048, [10, 10])


