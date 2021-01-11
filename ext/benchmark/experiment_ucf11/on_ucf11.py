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

import yaml

import torcherry as tc
from torcherry.utils.metric import MetricAccuracy, MetricLoss
from torcherry.utils.checkpoint import CheckBestValAcc
from torcherry.utils.util import set_env_seed

from model.cls_rnn import Classifier
from model.cls_rnn_tn import ClassifierTR, ClassifierBTT, ClassifierCP, ClassifierTT, ClassifierTK2


Models = dict(
    Classifier=Classifier,
    ClassifierTR=ClassifierTR,
    ClassifierBTT=ClassifierBTT,
    ClassifierCP=ClassifierCP,
    ClassifierTT=ClassifierTT,
    ClassifierTK2=ClassifierTK2,
)


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    set_env_seed(args.seed, use_cuda)

    dataset_path = os.path.join(proj_cfg.root_path, args.dataset)
    save_path = os.path.join(proj_cfg.save_root, "ucf11_feature", args.model, args.rank,
                             time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + "_%d" % args.seed)

    ranks = list(map(lambda x: int(x), args.rank.split(",")))

    if args.model in Models:
        model = Models[args.model](use_cuda=use_cuda, num_class=11, ranks=ranks,
                                   dataset_path=dataset_path, seed=args.seed, dropih=args.dropih)
    else:
        raise ValueError("model %s is not existed!" % args.model)

    # training_metrics = None
    training_metrics = [MetricAccuracy(1), MetricAccuracy(5), MetricLoss()]
    valing_metrics = [MetricAccuracy(1), MetricAccuracy(5), MetricLoss()]
    check_metrics = [CheckBestValAcc()]

    runner = tc.Runner(use_cuda)
    res_dict = runner.fit(model, save_path=save_path, train_epochs=args.epochs, train_callbacks=training_metrics,
               val_callbacks=valing_metrics, checkpoint_callbacks=check_metrics,
               continual_train_model_dir=args.continual_train_model_dir, pre_train_model_path=args.pre_train_model_path,
               record_setting="%s" % args,
               )

    with open(os.path.join(save_path, "info_record.yml"), "w", encoding="utf-8") as f:
        save_dict = dict(
            rank=ranks,
            acc=res_dict["acc"]
        )
        yaml.dump(save_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Working on UCF11 Feature")

    parser.add_argument("--dataset", type=str, default='./datasets/features')
    parser.add_argument("--model", type=str, default="ClassifierTR")
    parser.add_argument("--rank", type=str, default="40,60,48,48")
    parser.add_argument("--dropih", type=float, default=0.25)

    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument('--pre-train-model-path', type=str, default=None,
                        help='a pre-trained model path, set to None to disable the pre-training')
    parser.add_argument('--continual-train-model-dir', type=str, default=None,
                        help='continual training model folder, set to None to disable the keep training')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=233, help='normal random seed')

    args = parser.parse_args()
    print(args)

    main(args)
