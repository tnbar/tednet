# -*- coding: UTF-8 -*-

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torcherry as tc

import numpy as np

from .custom_dataset.cusprocess_ucf11 import UCF11Info, UCF11Torch

from .rnn_naive.rnn import NaiveLSTM

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])
Input_Size = np.prod([512, 2, 2])
Hidden_Size = 2048


class Classifier(tc.CherryModule):
    def __init__(self, dataset_path, seed, use_cuda, ranks=None, dropih=None, num_class=11):
        super(Classifier, self).__init__()
        self.dataset = UCF11Info(dataset_path, [512, 2, 2], 0.2, split_val=False, new_idx=False)
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        # self.weighted = 1. / torch.Tensor(self.dataset.weight_list).to(self.device)
        # self.weighted = self.weighted / torch.sum(self.weighted)

        self.input_size = Input_Size
        self.hidden_size = Hidden_Size

        self.lstm = NaiveLSTM(self.input_size, self.hidden_size)
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
        return MultiStepLR(optimizer, [100, 200], gamma=0.5)

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
            # sampler=train_sampler,
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


