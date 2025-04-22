from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        super().__init__(data_source)
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if not select_indexes:
                    continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])

        return iter(ret)


class RandomMultipleGallerySamplerNoCam(Sampler):
    def __init__(self, data_source, num_instances=4):
        super().__init__(data_source)

        self.data_source = data_source
        # 键为样本索引，值为对应的 pid（类别 ID）
        self.index_pid = defaultdict(int)
        # 键为 pid（类别 ID），值为对应的样本索引列表
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, _item in enumerate(data_source):
            pid = _item[1]
            # (_, pid, cam)
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_index[pid].append(index)

        # 所有类别 ID 的列表
        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        # 采样器将生成的总样本数，即类别数量乘以每个类别中的实例数量
        return self.num_samples * self.num_instances

    def __iter__(self):
        # 生成一个随机打乱的类别 ID 列表
        indices = torch.randperm(len(self.pids)).tolist()
        # 初始化返回列表 ret
        ret = []

        for kid in indices:
            # 从当前类别中随机选择一个样本索引 i
            i = random.choice(self.pid_index[self.pids[kid]])
            # _, i_pid, i_cam = self.data_source[i]
            # _, i_pid, i_cam
            ret.append(i)

            # index:获取该类别 ID 下的所有样本索引列表
            pid_i = self.index_pid[i]
            index = self.pid_index[pid_i]

            # 调用 No_index 函数，返回除 i 之外的样本索引列表
            select_indexes = No_index(index, i)

            # 如果没有可选择的样本，继续下一个循环
            if not select_indexes:
                continue
            # 根据 select_indexes 的长度选择是否重复采样
            if len(select_indexes) >= self.num_instances:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
            else:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

            # 将选择的样本索引添加到返回列表 ret 中
            for kk in ind_indexes:
                ret.append(index[kk])

        return iter(ret)
