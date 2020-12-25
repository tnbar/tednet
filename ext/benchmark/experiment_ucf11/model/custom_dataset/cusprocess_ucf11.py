# -*- coding: UTF-8 -*-

import os
import sys

import random
import pickle
from datetime import datetime

import numpy as np

from torchvision.datasets.vision import VisionDataset


classes = ['basketball', 'biking', 'diving', 'golf_swing', 'horse_riding', 'soccer_juggling',
           'swing', 'tennis_swing', 'trampoline_jumping', 'volleyball_spiking', 'walking']

total_frames = 0


class UCF11Torch(VisionDataset):
    def __init__(self, datainfo, mode, img_shape, transform=None, target_transform=None):
        super(UCF11Torch, self).__init__(root=None, transform=transform,
                                      target_transform=target_transform)

        self.img_shape = img_shape
        if mode == "train":
            self.data_list = datainfo.data_train
            self.label_list = datainfo.labels_train
        elif mode == "test":
            self.data_list = datainfo.data_test
            self.label_list = datainfo.labels_test
        else:
            raise TypeError("Mode of %s is not existed" % mode)

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        # Generate data
        file_path = self.data_list[index]
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        data = data.reshape(-1, *self.img_shape) / np.float32(128.)
        target = self.label_list[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.data_list)


class UCF11Info(object):
    def __init__(self, data_path, img_shape, test_rate, split_val=False, new_idx=False):
        self.data_path = data_path
        # self.max_time_len = int(max_time_len)
        self.img_shape = np.asarray(img_shape)
        assert len(self.img_shape) == 3

        self.test_rate = float(test_rate)
        self.split_val = split_val
        self.new_idx = new_idx
        # self.batch_size = int(batch_size)
        # self.shuffle = shuffle

        self.data_train, self.data_test, self.labels_train, self.labels_test = self.split()

    def split(self):
        if self.new_idx:
            val_rate = 0.1
            self.all_clips, self.all_labels = self.scan_clips()
            self.all_rel_clips = []
            for clip in self.all_clips:
                self.all_rel_clips.append(os.path.relpath(clip, self.data_path))

            datapairs_dict = dict()
            for i, label in enumerate(self.all_labels):
                if label in datapairs_dict.keys():
                    datapairs_dict[label].append(self.all_rel_clips[i])
                else:
                    datapairs_dict[label] = [self.all_rel_clips[i]]

            train_dict = dict()
            val_dict = dict()
            test_dict = dict()
            for key, value in datapairs_dict.items():
                class_length = len(value)
                test_num = int(class_length * self.test_rate)
                val_num = int(class_length * val_rate)
                train_num = class_length - test_num - val_num

                random.seed(datetime.now())
                random.shuffle(value)
                train_dict[key] = value[:train_num]
                val_dict[key] = value[train_num:(train_num+val_num)]
                test_dict[key] = value[(train_num+val_num):]

            index_dict = {"train": train_dict, "val": val_dict, "test": test_dict}

            with open(os.path.join(self.data_path, "indexes.pkl"), "wb") as f:
                pickle.dump(index_dict, f)

        else:
            with open(os.path.join(self.data_path, "indexes.pkl"), "rb") as f:
                index_dict = pickle.load(f)
                train_dict = index_dict["train"]
                val_dict = index_dict["val"]
                test_dict = index_dict["test"]

        self.count_dict = dict()
        if self.split_val:
            data_train = []
            labels_train = []
            for key, value in train_dict.items():
                self.count_dict[key] = len(value)
                for rel_path in value:
                    data_train.append(os.path.join(self.data_path, rel_path))
                    labels_train.append(key)

            data_test = []
            labels_test = []
            for key, value in val_dict.items():
                for rel_path in value:
                    data_test.append(os.path.join(self.data_path, rel_path))
                    labels_test.append(key)

        else:
            data_train = []
            labels_train = []
            for key, value in train_dict.items():
                self.count_dict[key] = len(value)
                for rel_path in value:
                    data_train.append(os.path.join(self.data_path, rel_path))
                    labels_train.append(key)

            for key, value in val_dict.items():
                self.count_dict[key] += len(value)
                for rel_path in value:
                    data_train.append(os.path.join(self.data_path, rel_path))
                    labels_train.append(key)

            data_test = []
            labels_test = []
            for key, value in test_dict.items():
                for rel_path in value:
                    data_test.append(os.path.join(self.data_path, rel_path))
                    labels_test.append(key)

        self.weight_list = []
        for i in range(11):
            self.weight_list.append(self.count_dict[i])
        # whole_count_dict = dict()
        # for label in labels:
        #     if label in whole_count_dict.keys():
        #         whole_count_dict[label] += 1
        #     else:
        #         whole_count_dict[label] = 1
        #
        # train_count_dict = dict()
        # for label in labels_train:
        #     if label in train_count_dict.keys():
        #         train_count_dict[label] += 1
        #     else:
        #         train_count_dict[label] = 1
        #
        # test_count_dict = dict()
        # for label in labels_test:
        #     if label in test_count_dict.keys():
        #         test_count_dict[label] += 1
        #     else:
        #         test_count_dict[label] = 1

        # rate_dict = dict()
        # for i in range(11):
        #     rate_dict[i] = test_count_dict[i] / train_count_dict[i]

        return data_train, data_test, labels_train, labels_test

    def scan_clips(self):
        res = []
        labels = []
        for ii, class_name in enumerate(classes):
            files = os.listdir(os.path.join(self.data_path, class_name))
            for this_collection in files:
                if this_collection == 'Annotation':
                    continue
                clips = os.listdir(os.path.join(self.data_path, class_name + '/' + this_collection))
                clips.sort()
                for this_clip in clips:
                    if not this_clip.endswith('pkl'):
                        continue
                    path = os.path.join(self.data_path, class_name + '/' + this_collection + '/' + this_clip)
                    res.append(path)
                    labels.append(ii)
        return np.asarray(res), np.asarray(labels)


# if __name__ == '__main__':
#     dataset = UCF11DataSet('/hdd/panyu/project/TR-RNN-PYTORCH/UCF11/datasets', 6, [160, 120, 3], 0.2, 128, True)
#     train_loader = iter(dataset.generate_train())
#     data = next(train_loader)
#     print(data)
#     pass
