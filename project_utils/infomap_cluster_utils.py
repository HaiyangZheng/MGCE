import numpy as np
# from tqdm import tqdm
import infomap
import faiss
import math
import multiprocessing as mp
import time

import torch
import collections


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
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


# 构造边
def get_links(single, links, nbrs, dists, min_sim):
    # for i in tqdm(range(nbrs.shape[0])):
    for i in range(nbrs.shape[0]):
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



def get_links_with_label_vectorized(single, links, nbrs, dists, min_sim, label_mark, if_labeled, args=None):
    suplinks_count = 0
    
    if args is not None and args.max_sim:
        n_samples = nbrs.shape[0]
        device = nbrs.device if hasattr(nbrs, 'device') else 'cpu'
        
        is_torch = hasattr(nbrs, 'device')
        
        if is_torch:
            nbrs_np = nbrs.cpu().numpy()
            dists_np = dists.cpu().numpy()
            label_mark_np = label_mark.cpu().numpy() if hasattr(label_mark, 'cpu') else np.array(label_mark)
            if_labeled_np = if_labeled.cpu().numpy() if hasattr(if_labeled, 'cpu') else np.array(if_labeled)
        else:
            nbrs_np = nbrs
            dists_np = dists
            label_mark_np = np.array(label_mark)
            if_labeled_np = np.array(if_labeled)
        
        i_indices = np.arange(n_samples)[:, np.newaxis]  # shape: (n_samples, 1)
        i_indices = np.broadcast_to(i_indices, nbrs_np.shape)  # shape: (n_samples, n_neighbors)
        
        rank_j_indices = np.arange(nbrs_np.shape[1])[np.newaxis, :]  # shape: (1, n_neighbors)
        rank_j_indices = np.broadcast_to(rank_j_indices, nbrs_np.shape)  # shape: (n_samples, n_neighbors)
        
        self_mask = i_indices != nbrs_np
        
        dist_mask = dists_np <= (1 - min_sim)
        
        valid_mask = self_mask & dist_mask
        
        valid_i = i_indices[valid_mask]
        valid_j = nbrs_np[valid_mask]
        valid_rank_j = rank_j_indices[valid_mask]
        valid_dists = dists_np[valid_mask]
        
        if len(valid_i) > 0:
            label_i = label_mark_np[valid_i]
            label_j = label_mark_np[valid_j]
            labeled_i = if_labeled_np[valid_i]
            labeled_j = if_labeled_np[valid_j]
            
            super_mask = (label_i == label_j) & labeled_i & labeled_j
            
            weights = 1.0 - valid_dists
            
            if super_mask.any():  
                super_indices = valid_i[super_mask]
                
                unique_i, inverse_indices = np.unique(super_indices, return_inverse=True)
                min_dists_per_i = np.maximum(dists_np[unique_i, 1], 0)  
                
                super_weights = 1.0 - min_dists_per_i[inverse_indices]
                weights[super_mask] = super_weights
                
                suplinks_count = int(super_mask.sum())
            
            for idx in range(len(valid_i)):
                links[(int(valid_i[idx]), int(valid_j[idx]))] = float(weights[idx])
        
        counts_per_node = np.zeros(n_samples, dtype=int)
        if len(valid_i) > 0:
            unique_nodes, counts = np.unique(valid_i, return_counts=True)
            counts_per_node[unique_nodes] = counts
        
        isolated_nodes = np.where(counts_per_node == 0)[0]
        single.extend(isolated_nodes.tolist())
        
        print("len of superlinks:", suplinks_count)
        
    return single, links




def calculate_cluster_similarity(cluster1_indices, cluster2_indices, feat_nbrs, feat_dists):
    """
    计算两个簇之间的相似度
    """
    if not cluster1_indices or not cluster2_indices:
        return 0.0
    
    cluster2_set = set(cluster2_indices)
    total_similarity = 0
    count = 0
    
    for idx1 in cluster1_indices:
        if idx1 >= len(feat_nbrs) or idx1 < 0:
            continue
            
        neighbors = feat_nbrs[idx1]
        distances = feat_dists[idx1]
        
        for i, neighbor_idx in enumerate(neighbors):
            if neighbor_idx in cluster2_set:
                similarity = 1 - distances[i]
                total_similarity += similarity
                count += 1
    
    return total_similarity / max(count, 1)

def find_best_merge_target(smallest_cluster_indices, other_clusters, feat_nbrs, feat_dists):
    """
    为最小簇找到最佳合并目标
    """
    if not other_clusters:
        return None
        
    best_similarity = -1
    best_target = None
    
    for target_label, target_indices in other_clusters:
        # 确保target_indices是列表或可迭代对象
        if not hasattr(target_indices, '__iter__'):
            print(f"警告: target_indices不是可迭代对象: {target_indices}")
            continue
            
        similarity = calculate_cluster_similarity(
            smallest_cluster_indices, target_indices, feat_nbrs, feat_dists
        )
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_target = (target_label, target_indices)
    
    return best_target

def merge_and_update(clusters, smallest_cluster_label, best_target_cluster):

    if best_target_cluster is None:
        result = []
        for item in clusters:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                print(f"警告: 发现格式错误的item: {item}")
                continue
            label, indices = item
            if label != smallest_cluster_label:
                result.append((label, indices))
        return result
    
    target_label, target_indices = best_target_cluster
    updated_clusters = []
    smallest_cluster_indices = None
    
    # 找到要被合并的最小簇
    for item in clusters:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            print(f"警告: 跳过格式错误的item: {item}")
            continue
        label, indices = item
        if label == smallest_cluster_label:
            smallest_cluster_indices = indices
            break
    
    if smallest_cluster_indices is None:
        print(f"警告: 未找到要合并的簇 {smallest_cluster_label}")
        return clusters
    
    for item in clusters:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        label, indices = item
        
        if label == smallest_cluster_label:
            continue  # 跳过最小簇
        elif label == target_label:
            # 合并索引
            merged_indices = list(indices) + list(smallest_cluster_indices)
            updated_clusters.append((label, merged_indices))
        else:
            updated_clusters.append((label, list(indices)))
    
    return updated_clusters

def merge_clusters(label2idx, feat_nbrs, feat_dists, target_num_clusters, verbose=False):

    if len(label2idx) <= target_num_clusters:
        if verbose:
            print(f"当前簇数量 {len(label2idx)} 已小于等于目标数量 {target_num_clusters}")
        result = [(label, indices) for label, indices in label2idx.items()]
        return result
    
    clusters = sorted(label2idx.items(), key=lambda x: len(x[1]))
    
    if verbose:
        print(f"开始合并: 当前 {len(clusters)} 个簇 -> 目标 {target_num_clusters} 个簇")
        print(f"最小簇大小: {len(clusters[0][1])}, 最大簇大小: {len(clusters[-1][1])}")
    
    iteration = 0
    start_time = time.time()
    
    while len(clusters) > target_num_clusters:
        iteration += 1
        
        if not clusters:
            print("错误: clusters为空")
            break
            
        # 检查第一个元素的格式
        first_item = clusters[0]
        if not isinstance(first_item, (list, tuple)) or len(first_item) != 2:
            print(f"错误: clusters[0]格式错误: {first_item}")
            break
            
        smallest_cluster_label, smallest_cluster_indices = first_item
        
        if verbose and iteration % 50 == 0:
            elapsed = time.time() - start_time
            print(f"迭代 {iteration}: 剩余 {len(clusters)} 个簇, "
                  f"正在合并簇 {smallest_cluster_label} (大小: {len(smallest_cluster_indices)})")
        
        # 找到最佳合并目标
        best_target_cluster = find_best_merge_target(
            smallest_cluster_indices, clusters[1:], feat_nbrs, feat_dists
        )
        
        if best_target_cluster is None:
            if verbose:
                print(f"警告: 无法为簇 {smallest_cluster_label} 找到合并目标")
        
        # 执行合并
        clusters = merge_and_update(clusters, smallest_cluster_label, best_target_cluster)
        
        # 检查合并后的格式
        if not clusters:
            print("错误: 合并后clusters为空")
            break
            
        # 重新排序
        try:
            clusters.sort(key=lambda x: len(x[1]) if isinstance(x, (list, tuple)) and len(x) >= 2 else 0)
        except Exception as e:
            print(f"排序错误: {e}")
            print(f"clusters内容: {clusters}")
            break
    
    if verbose:
        total_time = time.time() - start_time
        print(f"合并完成! 最终 {len(clusters)} 个簇, 总用时: {total_time:.2f}s")
        if clusters:
            print(f"最终簇大小分布: 最小={len(clusters[0][1])}, 最大={len(clusters[-1][1])}")
    
    return clusters

def clusters_to_predictions(clusters, total_samples):

    # 检查clusters的每个元素
    for i, item in enumerate(clusters):
        # print(f"  clusters[{i}]: type={type(item)}, value={item if len(str(item)) < 100 else str(item)[:100]+'...'}")
        if not isinstance(item, (list, tuple)):
            print(f"    ERROR: 不是tuple或list!")
        elif len(item) != 2:
            print(f"    ERROR: 长度不是2! 长度={len(item)}")
    
    predictions = np.full(total_samples, -1, dtype=int)
    
    for cluster_id, cluster_info in enumerate(clusters):
        # print(f"DEBUG: 处理cluster {cluster_id}, cluster_info类型: {type(cluster_info)}")
        
        # 强化类型检查
        if not isinstance(cluster_info, (list, tuple)):
            print(f"错误: cluster_info不是tuple或list: {cluster_info}")
            continue
            
        if len(cluster_info) != 2:
            print(f"错误: cluster_info长度不是2: {len(cluster_info)}, 内容: {cluster_info}")
            continue
        
        try:
            original_label, indices = cluster_info
            # print(f"  解包成功: label={original_label}, indices长度={len(indices) if hasattr(indices, '__len__') else 'N/A'}")
        except ValueError as e:
            print(f"错误: 无法解包cluster_info: {e}, 内容: {cluster_info}")
            continue
        
        # 处理indices
        if not hasattr(indices, '__iter__'):
            print(f"错误: indices不可迭代: {indices}")
            continue
            
        for idx in indices:
            try:
                idx = int(idx)
                if 0 <= idx < total_samples:
                    predictions[idx] = cluster_id
                else:
                    print(f"警告: 索引超出范围: {idx}")
            except (ValueError, TypeError) as e:
                print(f"警告: 无法转换索引: {idx}, 错误: {e}")
                continue
    
    return predictions


def cluster_by_semi_infomap(nbrs, dists, min_sim, cluster_num=2, label_mark=None, if_labeled=None, args=None, head=None, if_combine=False):
    """
    clustering based on Infomap
    :param nbrs:
    :param dists:
    :param pred_label_path:
    :return:
    """
    if label_mark is None:
        single = []
        links = {}
        with Timer('get links', verbose=True):
            single, links = get_links(single=single, links=links, nbrs=nbrs, dists=dists, min_sim=min_sim)

        infomapWrapper = infomap.Infomap("--two-level --directed")
        # for (i, j), sim in tqdm(links.items()):
        for (i, j), sim in links.items():
            _ = infomapWrapper.addLink(int(i), int(j), sim)

        infomapWrapper.run()

        label2idx = {}
        idx2label = {}

        for node in infomapWrapper.iterTree():
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


        for single_node in single:
            idx2label[single_node] = keys_len
            label2idx[keys_len] = [single_node]
            keys_len += 1
            node_count += 1


        idx_len = len(list(idx2label.keys()))
        assert idx_len == node_count, 'idx_len not equal node_count!'


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


        return pre_labels
    else:
        single = []
        links = {}
        with Timer('get links', verbose=True):

            single, links = get_links_with_label_vectorized(single=single, links=links, nbrs=nbrs, dists=dists, min_sim=min_sim, label_mark=label_mark, if_labeled=if_labeled, args=args)

        infomapWrapper = infomap.Infomap("--two-level --directed")
        # for (i, j), sim in tqdm(links.items()):
        for (i, j), sim in links.items():
            _ = infomapWrapper.addLink(int(i), int(j), sim)

        infomapWrapper.run()

        label2idx = {}
        idx2label = {}

        for node in infomapWrapper.iterTree():
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

        for single_node in single:
            idx2label[single_node] = keys_len
            label2idx[keys_len] = [single_node]
            keys_len += 1
            node_count += 1



        idx_len = len(list(idx2label.keys()))
        assert idx_len == node_count, 'idx_len not equal node_count!'

        if if_combine:
            clusters = merge_clusters(label2idx, nbrs, dists, args.mlp_out_dim)
            pre_labels1=clusters_to_predictions(clusters, len(nbrs))

            small_cluster = 0
            old_label_container = set()
            for each_label, each_index_list in label2idx.items():
                if len(each_index_list) <= cluster_num:
                    small_cluster += 1
                    for each_index in each_index_list:
                        idx2label[each_index] = -1
                else:
                    old_label_container.add(each_label)

            old2new = {old_label: new_label for new_label, old_label in enumerate(old_label_container)}

            for each_index, each_label in idx2label.items():
                if each_label == -1:
                    continue
                idx2label[each_index] = old2new[each_label]

            pre_labels2 = intdict2ndarray(idx2label)

            return pre_labels1, pre_labels2
        else:


            small_cluster = 0
            old_label_container = set()
            for each_label, each_index_list in label2idx.items():
                if len(each_index_list) <= cluster_num:
                    small_cluster += 1
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

            return pre_labels


def get_dist_nbr(features, k=80, knn_method='faiss-cpu', device=0):
    index = knn_faiss(feats=features, k=k, knn_method=knn_method, device=device)
    knns = index.get_knns()
    dists, nbrs = knns2ordered_nbrs(knns)
    return dists, nbrs


# generate new dataset and calculate cluster centers
@torch.no_grad()
def generate_cluster_features(labels, features, return_var=False):
    centers = collections.defaultdict(list)

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
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]
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

    def __init__(self, feats, k, knn_method='faiss-cpu', device=0, verbose=True):
        self.verbose = verbose

        with Timer('[{}] build index {}'.format(knn_method, k), verbose):
            feats = feats.astype('float32')
            size, dim = feats.shape
            if knn_method == 'faiss-gpu':
                i = math.ceil(size / 1000000)
                if i > 1:
                    i = (i - 1) * 4
                gpu_config = faiss.GpuIndexFlatConfig()
                gpu_config.device = device
                res = faiss.StandardGpuResources()
                res.setTempMemory(i * 1024 * 1024 * 1024)
                index = faiss.GpuIndexFlatIP(res, dim, gpu_config)
            else:
                index = faiss.IndexFlatIP(dim)
            index.add(feats)

        with Timer('[{}] query topk {}'.format(knn_method, k), verbose):
            sims, nbrs = index.search(feats, k=k)
            self.knns = [(np.array(nbr, dtype=np.int32),
                          1 - np.array(sim, dtype=np.float32))
                         for nbr, sim in zip(nbrs, sims)]

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
                    # tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
                    pool.imap(self.filter_by_th, range(tot)), total=tot)
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns



# 构造边
def get_links(single, links, nbrs, dists, min_sim):
    # for i in tqdm(range(nbrs.shape[0])):
    for i in range(nbrs.shape[0]):
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
    # for (i, j), sim in tqdm(links.items()):
    for (i, j), sim in links.items():
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

