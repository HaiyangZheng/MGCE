import numpy as np
from tqdm import tqdm
import infomap
import faiss
import math
import multiprocessing as mp
import time


class TextColors:
    HEADER = '\033[35m'
    OKBLUE = '\033[34m'
    OKGREEN = '\033[32m'
    WARNING = '\033[33m'
    FATAL = '\033[31m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name,
                time.time() - self.start))
        return exc_type is None




def l2norm(vec):
    """
    归一化
    :param vec:
    :return:
    """
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def read_meta(fn_meta, start_pos=0, verbose=True):
    """
    idx2lb：每一个顶点对应一个类
    lb2idxs：每个类对应一个id
    """
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb




def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        # 将knn转化为(num_samples, 2, k)形状的 NumPy 数组
        knns = np.array(knns)
    # 提取近邻索引并将其转换为 int32 类型
    nbrs = knns[:, 0, :].astype(np.int32)
    # 提取近邻距离
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        # 根据距离从低到高对每个样本的近邻进行排序
        nb_idx = np.argsort(dists, axis=1)
        # 根据排序索引重新排列距离和近邻索引
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


# 构造边
def get_links(single, links, nbrs, dists, min_sim):
    for i in tqdm(range(nbrs.shape[0])):
        count = 0
        for j in range(0, len(nbrs[i])):
            # 排除本身节点
            if i == nbrs[i][j]:
                pass
            elif dists[i][j] <= 1 - min_sim:
                count += 1
                links[(i, nbrs[i][j])] = float(1 - dists[i][j])
            else:
                break
        # 统计孤立点
        if count == 0:
            single.append(i)
    return single, links

import copy
# 构造边
def get_links_with_label(single, links, nbrs, dists, min_sim, label_mark, if_labeled, args=None):
    if args is not None and args.max_sim:# 控制属于有标签类别，且i和j属于一类，到底是赋值max_i还是1
        for i in tqdm(range(nbrs.shape[0])):
            count = 0
            for j in range(0, len(nbrs[i])):
                # 排除本身节点
                if i == nbrs[i][j]:
                    pass
                elif dists[i][j] <= 1 - min_sim:
                    count += 1
                    if label_mark[i] == label_mark[j] and if_labeled[i] == True:
                        # 属于有标签类别，且i和j属于一类，赋值为节点i和近邻的最大的相似性
                        links[(i, nbrs[i][j])] = float(1 - max(min(dists[i][1:]), 0))
                    else:
                        links[(i, nbrs[i][j])] = float(1 - dists[i][j])
                else:
                    break
            # 统计孤立点
            if count == 0:
                single.append(i)
        return single, links
        # for i in tqdm(range(nbrs.shape[0])):
        #     count = 0
        #     for j in range(0, len(nbrs[i])):
        #         # 排除本身节点
        #         if i == nbrs[i][j]:
        #             pass
        #         elif dists[i][j] <= 1 - min_sim:
        #             count += 1
        #             links[(i, nbrs[i][j])] = float(1 - dists[i][j])
        #         else:
        #             break
        #     # 统计孤立点
        #     if count == 0:
        #         single.append(i)
        # ref_link = copy.deepcopy(links)
        # for i in tqdm(range(nbrs.shape[0])):
        #     count = 0
        #     for j in range(0, len(nbrs[i])):
        #         # 排除本身节点
        #         if i == nbrs[i][j]:
        #             pass
        #         elif dists[i][j] <= 1 - min_sim:
        #             count += 1
        #             if label_mark[i] == label_mark[j] and if_labeled[i] == True:
        #                 links[(i, nbrs[i][j])] = max([ref_link[(i, pp)] for pp in nbrs[i, 1:]])
        #             else:
        #                 links[(i, nbrs[i][j])] = float(1 - dists[i][j])
        #         else:
        #             break
        #     # 统计孤立点
        #     if count == 0:
        #         single.append(i)
    else:
        for i in tqdm(range(nbrs.shape[0])):
            count = 0
            for j in range(0, len(nbrs[i])):
                # 排除本身节点
                if i == nbrs[i][j]:
                    pass
                elif dists[i][j] <= 1 - min_sim:
                    count += 1
                    if label_mark[i] == label_mark[j] and if_labeled[i] == True:
                        links[(i, nbrs[i][j])] = float(0.999999999)
                    else:
                        links[(i, nbrs[i][j])] = float(1 - dists[i][j])
                else:
                    break
            # 统计孤立点
            if count == 0:
                single.append(i)
        return single, links

def cluster_by_semi_infomap(nbrs, dists, min_sim, cluster_num=2, label_mark=None, if_labeled=None, args=None):
    """
    基于infomap的聚类
    :param nbrs:
    :param dists:
    :param pred_label_path:
    :return:
    """
    if label_mark is None:
        # single用于存储孤立点，links用于存储节点之间的链接及相似度
        single = []
        links = {}
        with Timer('get links', verbose=True):
            single, links = get_links(single=single, links=links, nbrs=nbrs, dists=dists, min_sim=min_sim)

        # 两阶段网络，有向图
        infomapWrapper = infomap.Infomap("--two-level --directed")
        for (i, j), sim in tqdm(links.items()):
            _ = infomapWrapper.addLink(int(i), int(j), sim)

        # 聚类运算
        infomapWrapper.run()

        # 两个字典用于存储聚类结果
        label2idx = {}
        idx2label = {}

        # 聚类结果统计
        # 遍历 Infomap 聚类结果树，获取label2idx
        # 输入伪标签簇的序号，得到所有特征向量编号
        for node in infomapWrapper.iterTree():
            # node.physicalId 特征向量的编号
            # node.moduleIndex() 聚类的编号
            if node.moduleIndex() not in label2idx:
                label2idx[node.moduleIndex()] = []
            label2idx[node.moduleIndex()].append(node.physicalId)

        node_count = 0
        # 过滤掉聚类 ID 为 0 的前两个节点和其他聚类 ID 的第一个节点
        # 更新idx2label
        # 输入idx，能得到伪标签序号
        for k, v in label2idx.items():
            if k == 0:
                each_index_list = v[2:]
                node_count += len(each_index_list)
                label2idx[k] = each_index_list
            else:
                each_index_list = v[1:]
                node_count += len(each_index_list)
                label2idx[k] = each_index_list

            for each_index in each_index_list:
                idx2label[each_index] = k

        keys_len = len(list(label2idx.keys()))
        # 孤立点放入到结果中
        # 将所有孤立点分配一个新的聚类 ID，并更新 label2idx 和 idx2label
        for single_node in single:
            idx2label[single_node] = keys_len
            label2idx[keys_len] = [single_node]
            keys_len += 1
            node_count += 1

        # 孤立点个数
        print("孤立点数：{}".format(len(single)))

        # 验证节点数量
        idx_len = len(list(idx2label.keys()))
        assert idx_len == node_count, 'idx_len not equal node_count!'

        print("总节点数：{}".format(idx_len))

        # 处理小簇
        old_label_container = set()
        for each_label, each_index_list in label2idx.items():
            # 将节点数小于等于 cluster_num 的簇标记为 -1
            if len(each_index_list) <= cluster_num:
                for each_index in each_index_list:
                    idx2label[each_index] = -1
            # 如果节点数多于cluster_num，那么old_label_container增加改簇
            else:
                old_label_container.add(each_label)

        # 将旧的簇 ID 映射到新的簇 ID
        old2new = {old_label: new_label for new_label, old_label in enumerate(old_label_container)}

        # 使用有效的簇ID更新 idx2label
        for each_index, each_label in idx2label.items():
            if each_label == -1:
                continue
            idx2label[each_index] = old2new[each_label]

        # 使用 intdict2ndarray 函数将 idx2label 转换为 NumPy 数组 pre_labels
        pre_labels = intdict2ndarray(idx2label)

        # 总类别数/有效类别数
        print("总类别数：{}/{}".format(keys_len, len(set(pre_labels)) - (1 if -1 in pre_labels else 0)))

        return pre_labels
    else:
        single = []
        links = {}
        with Timer('get links', verbose=True):
            single, links = get_links_with_label(single=single, links=links, nbrs=nbrs, dists=dists, min_sim=min_sim, label_mark=label_mark, if_labeled=if_labeled, args=args)

        infomapWrapper = infomap.Infomap("--two-level --directed")
        for (i, j), sim in tqdm(links.items()):
            _ = infomapWrapper.addLink(int(i), int(j), sim)

        # 聚类运算
        infomapWrapper.run()

        label2idx = {}
        idx2label = {}

        # 聚类结果统计
        for node in infomapWrapper.iterTree():
            # node.physicalId 特征向量的编号
            # node.moduleIndex() 聚类的编号
            if node.moduleIndex() not in label2idx:
                label2idx[node.moduleIndex()] = []
            label2idx[node.moduleIndex()].append(node.physicalId)

        node_count = 0
        for k, v in label2idx.items():
            if k == 0:
                each_index_list = v[2:]
                node_count += len(each_index_list)
                label2idx[k] = each_index_list
            else:
                each_index_list = v[1:]
                node_count += len(each_index_list)
                label2idx[k] = each_index_list

            for each_index in each_index_list:
                idx2label[each_index] = k

        keys_len = len(list(label2idx.keys()))
        # 孤立点放入到结果中
        for single_node in single:
            idx2label[single_node] = keys_len
            label2idx[keys_len] = [single_node]
            keys_len += 1
            node_count += 1

        # 孤立点个数
        print("孤立点数：{}".format(len(single)))

        idx_len = len(list(idx2label.keys()))
        assert idx_len == node_count, 'idx_len not equal node_count!'

        print("总节点数：{}".format(idx_len))

        old_label_container = set()
        for each_label, each_index_list in label2idx.items():
            if len(each_index_list) <= cluster_num:
                for each_index in each_index_list:
                    idx2label[each_index] = -1
            else:
                old_label_container.add(each_label)

        old2new = {old_label: new_label for new_label, old_label in enumerate(old_label_container)}

        for each_index, each_label in idx2label.items():
            if each_label == -1:
                continue
            idx2label[each_index] = old2new[each_label]

        pre_labels = intdict2ndarray(idx2label)

        print("总类别数：{}/{}".format(keys_len, len(set(pre_labels)) - (1 if -1 in pre_labels else 0)))

        return pre_labels


def get_dist_nbr(features, k=80, knn_method='faiss-cpu', device=0):
    index = knn_faiss(feats=features, k=k, knn_method=knn_method, device=device)
    knns = index.get_knns()
    dists, nbrs = knns2ordered_nbrs(knns)
    # (num_samples, k)形状
    return dists, nbrs

import torch
import collections

# generate new dataset and calculate cluster centers
@torch.no_grad()
# return_var决定是否返回每个簇的标准差
def generate_cluster_features(labels, features, return_var=False):
    #初始化centers字典
    # key：簇标签
    # value：属于该簇的特征向量列表
    centers = collections.defaultdict(list)

    # 遍历 labels，将每个特征向量 features[i] 添加到相应的聚类中心 centers[label[i]] 中
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])
    if return_var:
        centers = [
            torch.std_mean(torch.stack(centers[idx], dim=0), unbiased=True, dim=0) for idx in sorted(centers.keys())
        ]
        std, mean = [], []
        for _std, _mean in centers:
            std.append(_std)
            mean.append(_mean)
        mean = torch.stack(mean, dim=0)
        std = torch.stack(std, dim=0)
        return mean,  std

    else:
        # 对每个聚类的特征向量堆叠成一个张量，并计算均值
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]
        # 将所有聚类的均值堆叠成一个张量并返回
        centers = torch.stack(centers, dim=0)
        return centers





def l2norm(vec):
    """
    归一化
    :param vec:
    :return:
    """
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def read_meta(fn_meta, start_pos=0, verbose=True):
    """
    idx2lb：每一个顶点对应一个类
    lb2idxs：每个类对应一个id
    """
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


class knn_faiss():
    """
    内积暴力循环
    归一化特征的内积等价于余弦相似度
    """
    """
    feats：特征向量数组。
    k：要检索的最近邻数目。
    knn_method：选择使用 CPU 还是 GPU 进行计算（默认为 'faiss-cpu'）。
    device：如果使用 GPU，指定 GPU 设备 ID。
    verbose：是否打印详细信息。
    """

    def __init__(self, feats, k, knn_method='faiss-cpu', device=0, verbose=True):
        self.verbose = verbose

        with Timer('[{}] build index {}'.format(knn_method, k), verbose):
            # 特征向量转换为 float32 类型
            feats = feats.astype('float32')
            # 获取特征向量的数量和维度
            size, dim = feats.shape
            if knn_method == 'faiss-gpu':
                # 计算所需的 GPU 内存大小并配置 GPU
                i = math.ceil(size / 1000000)
                if i > 1:
                    i = (i - 1) * 4
                gpu_config = faiss.GpuIndexFlatConfig()
                gpu_config.device = device
                res = faiss.StandardGpuResources()
                res.setTempMemory(i * 1024 * 1024 * 1024)
                # 创建 GPU 版本的内积索引
                index = faiss.GpuIndexFlatIP(res, dim, gpu_config)
            else:
                # 默认方式：创建 CPU 版本的内积索引
                index = faiss.IndexFlatIP(dim)
            # 将特征向量添加到索引中
            index.add(feats)

        with Timer('[{}] query topk {}'.format(knn_method, k), verbose):
            # 使用 FAISS 的 search 方法检索每个特征向量的 K 个近邻，得到相似度和近邻索引
            sims, nbrs = index.search(feats, k=k)
            # 将近邻索引和相似度存储在 self.knns 中，并将相似度转换为距离（1 - 相似度）
            self.knns = [(np.array(nbr, dtype=np.int32),
                          1 - np.array(sim, dtype=np.float32))
                         for nbr, sim in zip(nbrs, sims)]

    # 根据距离阈值 th 过滤近邻
    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return th_nbrs, th_dists

    # 返回近邻，支持按阈值过滤
    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer('filter edges by th {} (CPU={})'.format(th, nproc),
                   self.verbose):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns



# 构造边
def get_links(single, links, nbrs, dists, min_sim):
    for i in tqdm(range(nbrs.shape[0])):
        count = 0
        for j in range(0, len(nbrs[i])):
            # 排除本身节点
            if i == nbrs[i][j]:
                pass
            elif dists[i][j] <= 1 - min_sim:
                count += 1
                links[(i, nbrs[i][j])] = float(1 - dists[i][j])
            else:
                break
        # 统计孤立点
        if count == 0:
            single.append(i)
    return single, links


def cluster_by_infomap(nbrs, dists, min_sim, cluster_num=2):
    """
    基于infomap的聚类
    :param nbrs:
    :param dists:
    :param pred_label_path:
    :return:
    """
    single = []
    links = {}
    with Timer('get links', verbose=True):
        single, links = get_links(single=single, links=links, nbrs=nbrs, dists=dists, min_sim=min_sim)

    infomapWrapper = infomap.Infomap("--two-level --directed")
    for (i, j), sim in tqdm(links.items()):
        _ = infomapWrapper.addLink(int(i), int(j), sim)

    # 聚类运算
    infomapWrapper.run()

    label2idx = {}
    idx2label = {}

    # 聚类结果统计
    for node in infomapWrapper.iterTree():
        # node.physicalId 特征向量的编号
        # node.moduleIndex() 聚类的编号
        if node.moduleIndex() not in label2idx:
            label2idx[node.moduleIndex()] = []
        label2idx[node.moduleIndex()].append(node.physicalId)

    node_count = 0
    for k, v in label2idx.items():
        if k == 0:
            each_index_list = v[2:]
            node_count += len(each_index_list)
            label2idx[k] = each_index_list
        else:
            each_index_list = v[1:]
            node_count += len(each_index_list)
            label2idx[k] = each_index_list

        for each_index in each_index_list:
            idx2label[each_index] = k

    keys_len = len(list(label2idx.keys()))
    # 孤立点放入到结果中
    for single_node in single:
        idx2label[single_node] = keys_len
        label2idx[keys_len] = [single_node]
        keys_len += 1
        node_count += 1

    # 孤立点个数
    print("孤立点数：{}".format(len(single)))

    idx_len = len(list(idx2label.keys()))
    assert idx_len == node_count, 'idx_len not equal node_count!'

    print("总节点数：{}".format(idx_len))

    old_label_container = set()
    for each_label, each_index_list in label2idx.items():
        if len(each_index_list) <= cluster_num:
            for each_index in each_index_list:
                idx2label[each_index] = -1
        else:
            old_label_container.add(each_label)

    old2new = {old_label: new_label for new_label, old_label in enumerate(old_label_container)}

    for each_index, each_label in idx2label.items():
        if each_label == -1:
            continue
        idx2label[each_index] = old2new[each_label]

    pre_labels = intdict2ndarray(idx2label)

    print("总类别数：{}/{}".format(keys_len, len(set(pre_labels)) - (1 if -1 in pre_labels else 0)))

    return pre_labels


#
# import torch
# import collections
#
# # generate new dataset and calculate cluster centers
# @torch.no_grad()
# def generate_cluster_features(labels, features):
#     centers = collections.defaultdict(list)
#
#     for i, label in enumerate(labels):
#         if label == -1:
#             continue
#         centers[labels[i]].append(features[i])
#
#     centers = [
#         torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
#     ]
#
#     centers = torch.stack(centers, dim=0)
#     return centers