# -*- coding: UTF-8 -*-

import sys
sys.path.append("../../..")

from managpu import GpuManager
my_gpu = GpuManager()
my_gpu.set_by_memory(1)

import os
import time
import argparse

from config import proj_cfg

import torch

import torcherry as tc
from torcherry.utils.metric import MetricAccuracy, MetricLoss
from torcherry.utils.checkpoint import CheckBestValAcc
from torcherry.utils.util import set_env_seed

from model.resnet_cifar10 import resnet20_cifar10, resnet32_cifar10
from model.resnet_tn_cifar10 import *

Models = dict(
    resnet20_cifar10=resnet20_cifar10,
    resnet32_cifar10=resnet32_cifar10,
    TRResNet20_CIFAR10=TRResNet20_CIFAR10,
    TRResNet32_CIFAR10=TRResNet32_CIFAR10,
    BTTResNet20_CIFAR10=BTTResNet20_CIFAR10,
    BTTResNet32_CIFAR10=BTTResNet32_CIFAR10,
    CPResNet20_CIFAR10=CPResNet20_CIFAR10,
    CPResNet32_CIFAR10=CPResNet32_CIFAR10,
    TTResNet20_CIFAR10=TTResNet20_CIFAR10,
    TTResNet32_CIFAR10=TTResNet32_CIFAR10,
    TK2ResNet20_CIFAR10=TK2ResNet20_CIFAR10,
    TK2ResNet32_CIFAR10=TK2ResNet32_CIFAR10,
)


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    set_env_seed(args.seed, use_cuda)

    dataset_path = os.path.join(proj_cfg.root_path, args.dataset)
    save_path = os.path.join(proj_cfg.save_root, "cifar10", args.model, args.rank,
                             time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + "_%d" % args.seed)

    ranks = list(map(lambda x: int(x), args.rank.split(",")))

    if args.model in Models:
        model = Models[args.model](rs=ranks, dataset_path=dataset_path, seed=args.seed)
    else:
        raise ValueError("model %s is not existed!" % args.model)

    # training_metrics = [MetricAccuracy(1), MetricLoss()]
    training_metrics = None
    valing_metrics = [MetricAccuracy(1), MetricAccuracy(5), MetricLoss()]
    check_metrics = [CheckBestValAcc()]

    runner = tc.Runner(use_cuda)
    res_dict = runner.fit(model, save_path=save_path, train_epochs=args.epochs, train_callbacks=training_metrics,
               val_callbacks=valing_metrics, checkpoint_callbacks=check_metrics,
               continual_train_model_dir=args.continual_train_model_dir, pre_train_model_path=args.pre_train_model_path,
               record_setting="%s" % args,
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Working on Cifar10")

    parser.add_argument("--dataset", type=str, default='./datasets/')
    parser.add_argument("--model", type=str, default="TRResNet32_CIFAR10")
    parser.add_argument("--rank", type=str, default="10,10,10,10,10,10,10")

    parser.add_argument("--epochs", type=int, default=200)

    parser.add_argument('--pre-train-model-path', type=str, default=None,
                        help='a pre-trained model path, set to None to disable the pre-training')
    parser.add_argument('--continual-train-model-dir', type=str, default=None,
                        help='continual training model folder, set to None to disable the keep training')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=233, help='normal random seed')

    args = parser.parse_args()
    print(args)

    main(args)
