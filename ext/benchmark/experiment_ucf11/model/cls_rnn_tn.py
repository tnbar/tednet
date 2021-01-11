# -*- coding: UTF-8 -*-

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torcherry as tc

import numpy as np
import tednet.tnn.tensor_ring as tr
import tednet.tnn.bt_tucker as btt
import tednet.tnn.cp as cp
import tednet.tnn.tensor_train as tt
import tednet.tnn.tucker2 as tk2

from .custom_dataset.cusprocess_ucf11 import UCF11Info, UCF11Torch

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])
Input_Size = np.prod([512, 2, 2])
Hidden_Size = 2048


class _ClassifierTN(tc.CherryModule):
    def __init__(self, dataset_path, seed, use_cuda, num_class=11):
        super(_ClassifierTN, self).__init__()
        self.dataset = UCF11Info(dataset_path, [512, 2, 2], 0.2, split_val=False, new_idx=False)
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.hidden_size = Hidden_Size

        self.fc = nn.Linear(self.hidden_size, num_class)

    def forward(self, x, state):
        input_shape = x.shape
        batch_size = input_shape[0]
        seq_size = input_shape[1]
        x = x.view(batch_size, seq_size, -1)
        x = x.permute(1, 0, 2)
        _, x = self.lstm(x, state)
        x = self.fc(x[0])
        return x

    def tc_train_step(self, model, data, target):
        batch_size = data.shape[0]
        state = LSTMState(torch.zeros(batch_size, self.hidden_size, device=self.device),
                          torch.zeros(batch_size, self.hidden_size, device=self.device))

        output_logits = model(data, state)

        loss = F.cross_entropy(output_logits, target)
        return loss

    def tc_val_step(self, model, data, target):
        batch_size = data.shape[0]
        state = LSTMState(torch.zeros(batch_size, self.hidden_size, device=self.device),
                          torch.zeros(batch_size, self.hidden_size, device=self.device))

        output_logits = model(data, state)

        loss = F.cross_entropy(output_logits, target)
        return output_logits, loss

    def tc_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4, weight_decay=0.00016667)

    def tc_lr_schedule(self, optimizer):
        return MultiStepLR(optimizer, [100, 200], gamma=0.2)

    def tc_train_loader(self):
        self.train_loader_type = "torchvision"
        train_loader = torch.utils.data.DataLoader(
            UCF11Torch(self.dataset, "train", [512, 2, 2],
                       # transform=transforms.Compose([
                       #         transforms.Normalize((), ()),
                       #         transforms.ToTensor(),
                       #     ])
                       ),
            batch_size=16, pin_memory=True, num_workers=0,
            shuffle=True,
        )
        return train_loader

    def tc_val_loader(self):
        self.val_loader_type = "torchvision"
        test_loader = torch.utils.data.DataLoader(
            UCF11Torch(self.dataset, "test", [512, 2, 2],
                       # transform=transforms.Compose([
                       #     transforms.ToTensor(),
                       # ])
                       ),
            batch_size=256, shuffle=False, pin_memory=True, num_workers=0)
        return test_loader


class ClassifierTR(_ClassifierTN):
    def __init__(self, dataset_path, seed, use_cuda, ranks, num_class=11, dropih=0.25):
        super(ClassifierTR, self).__init__(dataset_path=dataset_path, seed=seed, use_cuda=use_cuda, num_class=num_class)
        in_shape = [64, 32]
        hidden_shape = [32, 64]

        self.lstm = tr.TRLSTM(in_shape, hidden_shape, ranks, dropih)


class ClassifierBTT(_ClassifierTN):
    def __init__(self, dataset_path, seed, use_cuda, ranks, num_class=11, dropih=0.25):
        super(ClassifierBTT, self).__init__(dataset_path=dataset_path, seed=seed, use_cuda=use_cuda, num_class=num_class)
        in_shape = [64, 32]
        hidden_shape = [32, 64]

        self.lstm = btt.BTTLSTM(in_shape, hidden_shape, ranks, 2, dropih)


class ClassifierCP(_ClassifierTN):
    def __init__(self, dataset_path, seed, use_cuda, ranks, num_class=11, dropih=0.25):
        super(ClassifierCP, self).__init__(dataset_path=dataset_path, seed=seed, use_cuda=use_cuda, num_class=num_class)
        in_shape = [64, 32]
        hidden_shape = [32, 64]

        self.lstm = cp.CPLSTM(in_shape, hidden_shape, ranks[0], dropih)


class ClassifierTT(_ClassifierTN):
    def __init__(self, dataset_path, seed, use_cuda, ranks, num_class=11, dropih=0.25):
        super(ClassifierTT, self).__init__(dataset_path=dataset_path, seed=seed, use_cuda=use_cuda, num_class=num_class)
        in_shape = [64, 32]
        hidden_shape = [32, 64]

        self.lstm = tt.TTLSTM(in_shape, hidden_shape, ranks, dropih)


class ClassifierTK2(_ClassifierTN):
    def __init__(self, dataset_path, seed, use_cuda, ranks, num_class=11, dropih=0.25):
        super(ClassifierTK2, self).__init__(dataset_path=dataset_path, seed=seed, use_cuda=use_cuda, num_class=num_class)
        in_shape = [16, 16, 8]
        hidden_size = 2048

        self.lstm = tk2.TK2LSTM(in_shape, hidden_size, ranks, dropih)
