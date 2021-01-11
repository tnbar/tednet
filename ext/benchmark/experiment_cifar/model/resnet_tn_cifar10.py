# -*- coding: UTF-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim.lr_scheduler import MultiStepLR

import torcherry as tc
from torcherry.utils.scheduler import WarmMultiStepLR
from torcherry.dataset.dali import get_cifar10_iter_dali

import tednet.tnn.tensor_ring as tr
import tednet.tnn.bt_tucker as btt
import tednet.tnn.cp as cp
import tednet.tnn.tensor_train as tt
import tednet.tnn.tucker2 as tk2


class _Base_CIFAR10(tc.CherryModule):
    def tc_train_step(self, model, data, target):
        output_logits = model(data)
        loss = F.cross_entropy(output_logits, target)
        return loss

    def tc_val_step(self, model, data, target):
        output_logits = model(data)
        loss = F.cross_entropy(output_logits, target)
        return output_logits, loss

    def tc_optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=2e-2, momentum=0.9, weight_decay=5e-4)

    def tc_lr_schedule(self, optimizer):
        return MultiStepLR(optimizer, [60, 120, 160], 0.2)

    def tc_train_loader(self):
        self.train_loader_type = "dali"
        return get_cifar10_iter_dali(
        type='train', image_dir=self.data_path, batch_size=128, dali_cpu=False,
        num_threads=4, seed=self.seed, gpu_num=self.gpu_num,
    )

    def tc_val_loader(self):
        self.val_loader_type = "dali"
        return get_cifar10_iter_dali(
        type='val', image_dir=self.data_path, batch_size=128, gpu_num=self.gpu_num,
        num_threads=4, seed=self.seed, dali_cpu=False,
    )


class TRResNet20_CIFAR10(_Base_CIFAR10):
    def __init__(self, rs, dataset_path, seed=233):
        super(TRResNet20_CIFAR10, self).__init__()
        self.data_path = dataset_path
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.resnet = tr.TRResNet20(rs, 10)

    def forward(self, x):
        x = self.resnet(x)
        return x


class TRResNet32_CIFAR10(_Base_CIFAR10):
    def __init__(self, rs, dataset_path, seed=233):
        super(TRResNet32_CIFAR10, self).__init__()
        self.data_path = dataset_path
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.resnet = tr.TRResNet32(rs, 10)

    def forward(self, x):
        x = self.resnet(x)
        return x

class BTTResNet20_CIFAR10(_Base_CIFAR10):
    def __init__(self, rs, dataset_path, seed=233):
        super(BTTResNet20_CIFAR10, self).__init__()
        self.data_path = dataset_path
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.resnet = btt.BTTResNet20(rs, 10)

    def forward(self, x):
        x = self.resnet(x)
        return x


class BTTResNet32_CIFAR10(_Base_CIFAR10):
    def __init__(self, rs, dataset_path, seed=233):
        super(BTTResNet32_CIFAR10, self).__init__()
        self.data_path = dataset_path
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.resnet = btt.BTTResNet32(rs, 10)

    def forward(self, x):
        x = self.resnet(x)
        return x


class CPResNet20_CIFAR10(_Base_CIFAR10):
    def __init__(self, rs, dataset_path, seed=233):
        super(CPResNet20_CIFAR10, self).__init__()
        self.data_path = dataset_path
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.resnet = cp.CPResNet20(rs, 10)

    def forward(self, x):
        x = self.resnet(x)
        return x


class CPResNet32_CIFAR10(_Base_CIFAR10):
    def __init__(self, rs, dataset_path, seed=233):
        super(CPResNet32_CIFAR10, self).__init__()
        self.data_path = dataset_path
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.resnet = cp.CPResNet32(rs, 10)

        self.reset_parameters()

    def forward(self, x):
        x = self.resnet(x)
        return x

    def reset_parameters(self):
        for m in self.resnet.modules():
            if isinstance(m, cp.CPConv2D):
                for weight in m.parameters():
                    nn.init.normal_(weight.data, 0, 0.8)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, cp.CPLinear):
                for weight in m.parameters():
                    nn.init.normal_(weight.data, 0, 0.8)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class TTResNet20_CIFAR10(_Base_CIFAR10):
    def __init__(self, rs, dataset_path, seed=233):
        super(TTResNet20_CIFAR10, self).__init__()
        self.data_path = dataset_path
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.resnet = tt.TTResNet20(rs, 10)

    def forward(self, x):
        x = self.resnet(x)
        return x


class TTResNet32_CIFAR10(_Base_CIFAR10):
    def __init__(self, rs, dataset_path, seed=233):
        super(TTResNet32_CIFAR10, self).__init__()
        self.data_path = dataset_path
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.resnet = tt.TTResNet32(rs, 10)

    def forward(self, x):
        x = self.resnet(x)
        return x


class TK2ResNet20_CIFAR10(_Base_CIFAR10):
    def __init__(self, rs, dataset_path, seed=233):
        super(TK2ResNet20_CIFAR10, self).__init__()
        self.data_path = dataset_path
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.resnet = tk2.TK2ResNet20(rs, 10)

    def forward(self, x):
        x = self.resnet(x)
        return x


class TK2ResNet32_CIFAR10(_Base_CIFAR10):
    def __init__(self, rs, dataset_path, seed=233):
        super(TK2ResNet32_CIFAR10, self).__init__()
        self.data_path = dataset_path
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.resnet = tk2.TK2ResNet32(rs, 10)

    def forward(self, x):
        x = self.resnet(x)
        return x


