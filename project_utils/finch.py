import time
import torch
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
from sklearn.metrics import silhouette_score, fowlkes_mallows_score
import faiss
import multiprocessing as mp
from tqdm import tqdm
import math
from collections import defaultdict

pynndescent_available = False

def clust_rank(
    mat, use_ann_above_samples, 
    initial_rank=None, distance='cosine', verbose=False
):
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = np.empty(shape=(1, 1))
    elif s <= use_ann_above_samples:
        # generate pair-wise dist
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        # set self simi the largest distances
        np.fill_diagonal(orig_dist, 10)
        # find non-self closest instance
        initial_rank = np.argmin(orig_dist, axis=1)
        
    # The Clustering Equation
    # get the sp mat for the first-neighbor
    A = sp.csr_matrix(
        (
            np.ones_like(initial_rank, dtype=np.float32), 
            (np.arange(0, s), initial_rank)
        ), shape=(s, s)
    )
    # add it self
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    # make it sym, A (i to j), A.T (j to i), @ -- check if it is reciprocal
    A = A @ A.T
    A = A.tolil()
    # ignore self, 1 -- others link you, 2 -- double links
    A.setdiag(0)
    return A, orig_dist


def get_dist_nbr(features, k=80, knn_method='faiss-cpu', device=0):
    select_k = min(k, features.shape[0])
    index = knn_faiss(feats=features, k=select_k,
                      knn_method=knn_method, device=device)
    knns = index.get_knns()
    dists, nbrs = knns2ordered_nbrs(knns)
    dists, nbrs = torch.from_numpy(dists), torch.from_numpy(nbrs).long()
    # form a numpy array
    instance_count = dists.shape[0]
    out_dist = torch.from_numpy(
        10*np.ones((instance_count, instance_count))
    )
    out_dist.scatter_(1, nbrs, dists)
    return out_dist.numpy()


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


def get_clust(a, orig_dist, min_sim=None):
    # return pseudo labels and number of clusters
    if min_sim is not None:
        a[np.where((orig_dist * a) > min_sim)] = 0
    # get clust labels based on connections
    num_clust, u = sp.csgraph.connected_components(
        csgraph=a, directed=True, connection='weak', return_labels=True
    )
    return u, num_clust

def cool_mean(M, u):
    # filter out unlabeled data
    unlab = u!=-1
    M, u = M[unlab], u[unlab]
    # u -- labels
    s = M.shape[0]
    # un -- unique labels, nf -- counetr for each class
    un, nf = np.unique(u, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))
    # get centers
    return (umat.T @ M) / nf[..., np.newaxis]


def get_merge(c, u, data):
    # u--labels, data -- inputs
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig] # reformulate the label space
    else:
        # this way for the first round of clustering
        c = u
    mat = cool_mean(data, c)
    return c, mat # pse labels and centers


def FINCH(data, adj=None, orig_dist = None, initial_rank=None, 
          ensure_early_exit=True, verbose=True, 
          use_ann_above_samples=70000):
    # Cast input data to float32
    data, min_sim = data.astype(np.float32), None
    # adj -- (neighbor sparse mat), orig_dist -- (distance matrix), further merge
    if adj is None:
        adj, orig_dist = clust_rank(
            data, use_ann_above_samples, 
            initial_rank, 'cosine'
        )
        
    # get current clustering results based on adj
    group, num_clust = get_clust(adj, [], min_sim)
    # pse labels and formulated cenetrs
    c, mat = get_merge([], group, data)
    
    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj)

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust, ]
    # mat -- feature center
    while exit_clust > 1:
        # repeat the previous process, use previous centers for init
        adj, orig_dist = clust_rank(
            mat, use_ann_above_samples, initial_rank, 'cosine'
        )
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        # pseudo labels and update new centers
        c_, mat = get_merge(c_, u, data)

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr
        
        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1
        
    return c


class knn_faiss():
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
                # gpu_config.device = device
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


if __name__ == '__main__':
    data = np.random.rand(
        10, 200
    ).astype(np.float32)
    c = FINCH(
        data, initial_rank=None, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True
    )[0]