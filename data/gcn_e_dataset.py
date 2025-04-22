import numpy as np
from tqdm import tqdm

from .utils import (read_meta, l2norm, build_knns,
                   knns2ordered_nbrs, fast_knns2spmat, row_normalize,
                   build_symmetric_adj, Timer, density)

import torch



class KNNGraphDataset(object):
    def __init__(self, features, labels=None,
                 knn=10, feature_dim=768, is_norm=True,
                 th_sim=0, max_conn=1, conf_metric=False, ignore_ratio=0.8,
                 ignore_small_confs=True, use_candidate_set=True, radius=0.3):
        self.feature_dim = feature_dim
        self.is_norm_feat = is_norm
        self.labels = labels
        self.k = knn
        self.th_sim = th_sim
        self.max_conn = max_conn
        self.ignore_ratio = ignore_ratio
        self.ignore_small_confs = ignore_small_confs
        self.use_candidate_set = use_candidate_set
        self.max_qsize = 1e5
        self.conf_metric = conf_metric
        self.features = features

        with Timer('read meta and feature'):
            if labels is not None:
                # for training with labels
                self.lb2idxs, self.idx2lb = read_meta(labels)
                # get num of verts
                self.inst_num = len(self.idx2lb)
                self.ignore_label = False
            else:
                self.inst_num = -1
                self.ignore_label = True

            if self.is_norm_feat:
                self.features = l2norm(self.features)
            if self.inst_num == -1:
                self.inst_num = self.features.shape[0]
            self.size = 1

        print('feature shape: {}, k: {}, norm_feat: {}'.format(
            self.features.shape, self.k, self.is_norm_feat))

        with Timer('read knn graph'):
            knn_prefix = 'knn'
            # get KNN for each node, topk with the highest cos simi
            knns = build_knns(
                knn_prefix, self.features, 'faiss',
                self.k, is_rebuild=True
            )
            # get adj matrix, contains "distance" (not simi)
            adj = fast_knns2spmat(knns, self.k, self.th_sim, use_sim=True)
            # build symmetric adjacency matrix
            adj = build_symmetric_adj(adj, self_loop=True)
            self.adj = row_normalize(adj)

            # convert knns to (dists, nbrs)
            self.dists, self.nbrs = knns2ordered_nbrs(knns, sort=True)
            # use self func to estimate confi
            self.confs = density(self.dists, radius=radius)

            assert 0 <= self.ignore_ratio <= 1
            if self.ignore_ratio == 1:
                self.ignore_set = set(np.arange(len(self.confs)))
            else:
                # ignore part of the samples
                num = int(self.dists.shape[0] * self.ignore_ratio)
                # gonna to ignore points with low confs
                confs = self.confs
                # predict sample links for low conf nodes
                if not self.ignore_small_confs: confs = -confs
                self.ignore_set = set(np.argpartition(confs, num)[:num])

        print('ignore_ratio: {}, ignore_small_confs: {}, use_candidate_set: {}'.
              format(self.ignore_ratio, self.ignore_small_confs, self.use_candidate_set))
        print('#ignore_set: {} / {} = {:.3f}'.format(
            len(self.ignore_set), self.inst_num, len(self.ignore_set) / self.inst_num))

        with Timer('Prepare sub-graphs'):
            # construct subgraphs with larger confidence
            self.peaks = {i: [] for i in range(self.inst_num)}
            self.dist2peak = {i: [] for i in range(self.inst_num)}

            # find top knn for each inst, if it is in ignore_set, just use KNN
            results = [
                self.get_subgraph(i) for i in tqdm(range(self.inst_num))
            ]

            self.adj_lst, self.feat_lst, self.lb_lst = [], [], []
            self.subset_gt_labels, self.subset_idxs = [], []
            self.subset_nbrs, self.subset_dists = [], []
            # relabel each sub graph
            for result in results:
                if result is None:
                    continue
                elif len(result) == 3:
                    # for samples in ignore set
                    i, nbr, dist = result
                    # use 1NN as the final pred of pos samples
                    self.peaks[i].extend(nbr)
                    self.dist2peak[i].extend(dist)
                    continue

                # this way, for samples need to be pred by GCN
                i, nbr, dist, feat, adj, lb = result
                # ind for those need to be pred by GCN
                self.subset_idxs.append(i)
                # i's neighbour
                self.subset_nbrs.append(nbr)
                self.subset_dists.append(dist)
                self.feat_lst.append(feat)
                self.adj_lst.append(adj)

                if not self.ignore_label:
                    # generate subg label
                    self.subset_gt_labels.append(self.idx2lb[i])
                    self.lb_lst.append(lb)
            # pids for all sub_g anchors
            self.subset_gt_labels = np.array(self.subset_gt_labels)

            self.size = len(self.feat_lst)
            assert self.size == len(self.adj_lst)
            if not self.ignore_label:
                assert self.size == len(self.lb_lst)

    def get_subgraph(self, i):
        nbr = self.nbrs[i]  # neighbors, including self
        dist = self.dists[i]  # including self
        # find neighbors that have higher confis, filter out self idx based on nbr
        idxs = range(1, len(self.confs[nbr]))  # np.where(self.confs[nbr] > self.confs[i])[0]

        if len(idxs) == 0:
            return None
        elif len(idxs) == 1 or i in self.ignore_set:
            # ignored samples will use KNN to construct graph, not the GCN-predicted ones
            nbr_lst, dist_lst = [], []
            for j in idxs[:self.max_conn]:
                # get top-max_conn
                nbr_lst.append(nbr[j])
                dist_lst.append(self.dists[i, j])
            return i, nbr_lst, dist_lst

        # this way
        if self.use_candidate_set:
            # obtain those samples in candi, those samples are of high confidence
            nbr, dist = nbr[idxs], dist[idxs]

        # present `direction`, use samples not in ignore set to pred
        feat = self.features[nbr] - self.features[i]  # deduct itself
        # sample small graph from the whole, form nei-wise graph
        adj = self.adj[nbr, :][:, nbr]
        adj = row_normalize(adj).astype(np.float32)

        if not self.ignore_label:
            # this way, check if the nbr is pos for anchor
            lb = [int(self.idx2lb[i] == self.idx2lb[n]) for n in nbr]
        else:
            lb = [0 for _ in nbr]  # dummy labels
        # whether it has the same pid as the i
        lb = np.array(lb)

        return i, nbr, dist, feat, adj, lb

    def __getitem__(self, index):
        features = self.feat_lst[index]
        adj = self.adj_lst[index]
        if not self.ignore_label:
            labels = self.lb_lst[index]
        else:
            labels = -1

        return torch.from_numpy(features), torch.from_numpy(adj.toarray()).float(), torch.from_numpy(labels)


    def __len__(self):
        # only use chosen samples for training
        return self.size


class GCNEDataset(object):
    def __init__(self, features, labels=None, 
                 knn=10, feature_dim=768, is_norm=True,
                 th_sim=0, max_conn=1, conf_metric=False, ignore_ratio=0.8,
                 ignore_small_confs=True, use_candidate_set=True, radius=0.3):
        self.feature_dim = feature_dim
        self.is_norm_feat = is_norm
        self.labels = labels
        self.k = knn
        self.th_sim = th_sim
        self.max_conn = max_conn
        self.ignore_ratio = ignore_ratio
        self.ignore_small_confs = ignore_small_confs
        self.use_candidate_set = use_candidate_set
        self.max_qsize = 1e5
        self.conf_metric = conf_metric
        self.features = features
        
        with Timer('read meta and feature'):
            if labels is not None:
                # for training with labels
                self.lb2idxs, self.idx2lb = read_meta(labels)
                # get num of verts
                self.inst_num = len(self.idx2lb)
                self.ignore_label = False
            else:
                self.inst_num = -1
                self.ignore_label = True
            
            if self.is_norm_feat:
                self.features = l2norm(self.features)
            if self.inst_num == -1:
                self.inst_num = self.features.shape[0]
            self.size = 1 

        print('feature shape: {}, k: {}, norm_feat: {}'.format(
            self.features.shape, self.k, self.is_norm_feat))
        
        with Timer('read knn graph'):
            knn_prefix = 'knn'
            # get KNN for each node, topk with the highest cos simi
            knns = build_knns(
                knn_prefix, self.features, 'faiss', 
                self.k, is_rebuild=True
            )
            # get adj matrix, contains "distance" (not simi)
            adj = fast_knns2spmat(knns, self.k, self.th_sim, use_sim=True)
            # build symmetric adjacency matrix
            adj = build_symmetric_adj(adj, self_loop=True)
            self.adj = row_normalize(adj)

            # convert knns to (dists, nbrs)
            self.dists, self.nbrs = knns2ordered_nbrs(knns, sort=True)
            # use self func to estimate confi
            self.confs = density(self.dists, radius=radius)

            assert 0 <= self.ignore_ratio <= 1
            if self.ignore_ratio == 1:
                self.ignore_set = set(np.arange(len(self.confs)))
            else:
                # ignore part of the samples
                num = int(self.dists.shape[0] * self.ignore_ratio)
                # gonna to ignore points with low confs
                confs = self.confs
                # predict sample links for low conf nodes
                if not self.ignore_small_confs: confs = -confs
                self.ignore_set = set(np.argpartition(confs, num)[:num])

        print('ignore_ratio: {}, ignore_small_confs: {}, use_candidate_set: {}'.
            format(self.ignore_ratio, self.ignore_small_confs, self.use_candidate_set))
        print('#ignore_set: {} / {} = {:.3f}'.format(
            len(self.ignore_set), self.inst_num, len(self.ignore_set) / self.inst_num))

        with Timer('Prepare sub-graphs'):
            # construct subgraphs with larger confidence
            self.peaks = {i: [] for i in range(self.inst_num)}
            self.dist2peak = {i: [] for i in range(self.inst_num)}

            # find top knn for each inst, if it is in ignore_set, just use KNN
            results = [
                self.get_subgraph(i) for i in tqdm(range(self.inst_num))
            ]

            self.adj_lst, self.feat_lst, self.lb_lst = [], [], []
            self.subset_gt_labels, self.subset_idxs = [], []
            self.subset_nbrs, self.subset_dists = [], []
            # relabel each sub graph
            for result in results:
                if result is None: continue
                elif len(result) == 3:
                    # for samples in ignore set
                    i, nbr, dist = result
                    # use 1NN as the final pred of pos samples
                    self.peaks[i].extend(nbr)
                    self.dist2peak[i].extend(dist)
                    continue
                
                # this way, for samples need to be pred by GCN
                i, nbr, dist, feat, adj, lb = result
                # ind for those need to be pred by GCN
                self.subset_idxs.append(i)
                # i's neighbour
                self.subset_nbrs.append(nbr)
                self.subset_dists.append(dist)
                self.feat_lst.append(feat)
                self.adj_lst.append(adj)
                
                if not self.ignore_label:
                    # generate subg label
                    self.subset_gt_labels.append(self.idx2lb[i])
                    self.lb_lst.append(lb)
            # pids for all sub_g anchors
            self.subset_gt_labels = np.array(self.subset_gt_labels)

            self.size = len(self.feat_lst)
            assert self.size == len(self.adj_lst)
            if not self.ignore_label:
                assert self.size == len(self.lb_lst)

    def get_subgraph(self, i):
        nbr = self.nbrs[i] # neighbors, including self
        dist = self.dists[i] # including self
        # find neighbors that have higher confis, filter out self idx based on nbr
        idxs = range(1, len(self.confs[nbr])) #np.where(self.confs[nbr] > self.confs[i])[0]

        if len(idxs) == 0: return None
        elif len(idxs) == 1 or i in self.ignore_set:
            # ignored samples will use KNN to construct graph, not the GCN-predicted ones
            nbr_lst, dist_lst = [], []
            for j in idxs[:self.max_conn]:
                # get top-max_conn
                nbr_lst.append(nbr[j])
                dist_lst.append(self.dists[i, j])
            return i, nbr_lst, dist_lst

        # this way
        if self.use_candidate_set:
            # obtain those samples in candi, those samples are of high confidence
            nbr, dist = nbr[idxs], dist[idxs]

        # present `direction`, use samples not in ignore set to pred
        feat = self.features[nbr] - self.features[i] # deduct itself
        # sample small graph from the whole, form nei-wise graph
        adj = self.adj[nbr, :][:, nbr]
        adj = row_normalize(adj).astype(np.float32)

        if not self.ignore_label:
            # this way, check if the nbr is pos for anchor
            lb = [int(self.idx2lb[i] == self.idx2lb[n]) for n in nbr]
        else:
            lb = [0 for _ in nbr]  # dummy labels
        # whether it has the same pid as the i
        lb = np.array(lb)

        return i, nbr, dist, feat, adj, lb

    def __getitem__(self, index):
        features = self.feat_lst[index]
        adj = self.adj_lst[index]
        if not self.ignore_label:
            labels = self.lb_lst[index]
        else:
            labels = -1
        return features, adj, labels

    def __len__(self):
        # only use chosen samples for training
        return self.size
