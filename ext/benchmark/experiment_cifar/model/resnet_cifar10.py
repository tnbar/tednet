# -*- coding: UTF-8 -*-

'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim.lr_scheduler import MultiStepLR

import torcherry as tc
from torcherry.utils.scheduler import WarmMultiStepLR

from torcherry.dataset.dali import get_cifar10_iter_dali  # use original data


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_CIFAR10(tc.CherryModule):
    def __init__(self, block, num_blocks, dataset_path, seed=233, num_classes=10):
        super(ResNet_CIFAR10, self).__init__()
        self.data_path = dataset_path
        self.seed = seed
        self.gpu_num = torch.cuda.device_count()

        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

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


def resnet20_cifar10(dataset_path, seed=233, rs=None):
    return ResNet_CIFAR10(BasicBlock, [3, 3, 3], dataset_path, seed=seed)


def resnet32_cifar10(dataset_path, seed=233, rs=None):
    return ResNet_CIFAR10(BasicBlock, [5, 5, 5], dataset_path, seed=seed)

