from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length is not None:
            return self.length

        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


import os.path as osp
from torch.utils.data import DataLoader, Dataset

from PIL import Image

# # 原版的代码
# class FakeLabelDataset(Dataset):
#     def __init__(self, dataset, root=None, transform=None):
#         super(FakeLabelDataset, self).__init__()
#         self.dataset = dataset
#         self.root = root
#         self.transform = transform
#         try:
#             if isinstance(self.dataset[0][0], str):
#                 self.data0_is_numpy = False
#             else:
#                 self.data0_is_numpy = True
#         except:
#             self.data0_is_numpy = False
#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, indices):
#         return self._get_single_item(indices)

#     def _get_single_item(self, index):
#         if self.data0_is_numpy:
#             # image_numpy, fake, real = self.dataset[index]
#             item_ = self.dataset[index]
#             image_numpy, fake, real = item_[0], item_[1], item_[2]

#             img = Image.fromarray(image_numpy.astype('uint8')).convert('RGB')
#             if self.transform is not None:
#                 img = self.transform(img)

#             return img, fake, real

#         else:
#             # fname, fake, real = self.dataset[index]
#             item_ = self.dataset[index]
#             fname, fake, real = item_[0], item_[1], item_[2]
#             fpath = fname
#             if self.root is not None:
#                 fpath = osp.join(self.root, fname)

#             img = Image.open(fpath).convert('RGB')

#             if self.transform is not None:
#                 img = self.transform(img)

#             return img, fake, real

# 原版的代码注释版
class FakeLabelDataset(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(FakeLabelDataset, self).__init__()
        # 3个基本数据集元素
        self.dataset = dataset
        self.root = root
        self.transform = transform
        # 如果第一个元素的第 0 项是 str，就认为是路径；否则认为是 numpy 图像
        try:
            if isinstance(self.dataset[0][0], str):
                self.data0_is_numpy = False
            else:
                self.data0_is_numpy = True
        except:
            self.data0_is_numpy = False

    # 返回样本数
    def __len__(self):
        return len(self.dataset)

    # 直接转调下一个函数
    def __getitem__(self, indices):
        return self._get_single_item(indices)
    

    # 内存已有 numpy 的情况， 把 numpy 转成 PIL.Image，再转 RGB，再做 transform
    def _get_single_item(self, index):
        if self.data0_is_numpy:
            # image_numpy, fake, real = self.dataset[index]
            item_ = self.dataset[index]
            image_numpy, fake, real = item_[0], item_[1], item_[2]

            img = Image.fromarray(image_numpy.astype('uint8')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            return img, fake, real

        else:
            # fname, fake, real = self.dataset[index]
            item_ = self.dataset[index]
            fname, fake, real = item_[0], item_[1], item_[2]
            fpath = fname
            if self.root is not None:
                fpath = osp.join(self.root, fname)

            img = Image.open(fpath).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            return img, fake, real

class FakeLabelDataset_3head(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(FakeLabelDataset_3head, self).__init__()
        # 3个基本数据集元素
        self.dataset = dataset
        self.root = root
        self.transform = transform
        # 如果第一个元素的第 0 项是 str，就认为是路径；否则认为是 numpy 图像
        try:
            if isinstance(self.dataset[0][0], str):
                self.data0_is_numpy = False
            else:
                self.data0_is_numpy = True
        except:
            self.data0_is_numpy = False

    # 返回样本数
    def __len__(self):
        return len(self.dataset)

    # 直接转调下一个函数
    def __getitem__(self, indices):
        return self._get_single_item(indices)
    

    # 内存已有 numpy 的情况， 把 numpy 转成 PIL.Image，再转 RGB，再做 transform
    def _get_single_item(self, index):
        if self.data0_is_numpy:
            # image_numpy, fake, real = self.dataset[index]
            item_ = self.dataset[index]
            image_numpy, dccl, dccl2, dccl3, real = item_[0], item_[1], item_[2], item_[3], item_[4]
            # image_numpy, dccl, dccl2, dccl3, real, if_labeled = item_[0], item_[1], item_[2], item_[3], item_[4], item_[5]

            img = Image.fromarray(image_numpy.astype('uint8')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            # return img, dccl, dccl2, dccl3, real, if_labeled
            return img, dccl, dccl2, dccl3, real

        else:
            # fname, fake, real = self.dataset[index]
            item_ = self.dataset[index]
            fname, dccl, dccl2, dccl3, real = item_[0], item_[1], item_[2], item_[3], item_[4]
            # fname, dccl, dccl2, dccl3, real, if_labeled = item_[0], item_[1], item_[2], item_[3], item_[4], item_[5]
            fpath = fname
            if self.root is not None:
                fpath = osp.join(self.root, fname)

            img = Image.open(fpath).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            # return img, dccl, dccl2, dccl3, real, if_labeled
            return img, dccl, dccl2, dccl3, real

class FakeLabelDataset_2head(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(FakeLabelDataset_2head, self).__init__()
        # 3个基本数据集元素
        self.dataset = dataset
        self.root = root
        self.transform = transform
        # 如果第一个元素的第 0 项是 str，就认为是路径；否则认为是 numpy 图像
        try:
            if isinstance(self.dataset[0][0], str):
                self.data0_is_numpy = False
            else:
                self.data0_is_numpy = True
        except:
            self.data0_is_numpy = False

    # 返回样本数
    def __len__(self):
        return len(self.dataset)

    # 直接转调下一个函数
    def __getitem__(self, indices):
        return self._get_single_item(indices)
    

    # 内存已有 numpy 的情况， 把 numpy 转成 PIL.Image，再转 RGB，再做 transform
    def _get_single_item(self, index):
        if self.data0_is_numpy:
            # image_numpy, fake, real = self.dataset[index]
            item_ = self.dataset[index]
            image_numpy, dccl, dccl2, real = item_[0], item_[1], item_[2], item_[3]
            # image_numpy, dccl, dccl2, dccl3, real, if_labeled = item_[0], item_[1], item_[2], item_[3], item_[4], item_[5]

            img = Image.fromarray(image_numpy.astype('uint8')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            # return img, dccl, dccl2, dccl3, real, if_labeled
            return img, dccl, dccl2, real

        else:
            # fname, fake, real = self.dataset[index]
            item_ = self.dataset[index]
            fname, dccl, dccl2, real = item_[0], item_[1], item_[2], item_[3]
            # fname, dccl, dccl2, dccl3, real, if_labeled = item_[0], item_[1], item_[2], item_[3], item_[4], item_[5]
            fpath = fname
            if self.root is not None:
                fpath = osp.join(self.root, fname)

            img = Image.open(fpath).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            # return img, dccl, dccl2, dccl3, real, if_labeled
            return img, dccl, dccl2, real


class FakeLabelDataset_4head(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(FakeLabelDataset_4head, self).__init__()
        # 3个基本数据集元素
        self.dataset = dataset
        self.root = root
        self.transform = transform
        # 如果第一个元素的第 0 项是 str，就认为是路径；否则认为是 numpy 图像
        try:
            if isinstance(self.dataset[0][0], str):
                self.data0_is_numpy = False
            else:
                self.data0_is_numpy = True
        except:
            self.data0_is_numpy = False

    # 返回样本数
    def __len__(self):
        return len(self.dataset)

    # 直接转调下一个函数
    def __getitem__(self, indices):
        return self._get_single_item(indices)
    

    # 内存已有 numpy 的情况， 把 numpy 转成 PIL.Image，再转 RGB，再做 transform
    def _get_single_item(self, index):
        if self.data0_is_numpy:
            # image_numpy, fake, real = self.dataset[index]
            item_ = self.dataset[index]
            image_numpy, dccl, dccl2, dccl3, dccl4, real = item_[0], item_[1], item_[2], item_[3], item_[4], item_[5]
            # image_numpy, dccl, dccl2, dccl3, real, if_labeled = item_[0], item_[1], item_[2], item_[3], item_[4], item_[5]

            img = Image.fromarray(image_numpy.astype('uint8')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            # return img, dccl, dccl2, dccl3, real, if_labeled
            return img, dccl, dccl2, dccl3, dccl4, real

        else:
            # fname, fake, real = self.dataset[index]
            item_ = self.dataset[index]
            fname, dccl, dccl2, dccl3, dccl4, real = item_[0], item_[1], item_[2], item_[3], item_[4], item_[5]
            # fname, dccl, dccl2, dccl3, real, if_labeled = item_[0], item_[1], item_[2], item_[3], item_[4], item_[5]
            fpath = fname
            if self.root is not None:
                fpath = osp.join(self.root, fname)

            img = Image.open(fpath).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            # return img, dccl, dccl2, dccl3, real, if_labeled
            return img, dccl, dccl2, dccl3, dccl4, real


class FakeLabelDataset_with_trueindex(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(FakeLabelDataset_with_trueindex, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        try:
            if isinstance(self.dataset[0][0], str):
                self.data0_is_numpy = False
            else:
                self.data0_is_numpy = True
        except:
            self.data0_is_numpy = False
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        if self.data0_is_numpy:
            # image_numpy, fake, real = self.dataset[index]
            item_ = self.dataset[index]
            image_numpy, fake, real, true_index = item_[0], item_[1], item_[2], item_[4]

            img = Image.fromarray(image_numpy.astype('uint8')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            return img, fake, real, true_index

        else:
            # fname, fake, real = self.dataset[index]
            item_ = self.dataset[index]
            fname, fake, real, true_index = item_[0], item_[1], item_[2], item_[4]
            fpath = fname
            if self.root is not None:
                fpath = osp.join(self.root, fname)

            img = Image.open(fpath).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            return img, fake, real, true_index
