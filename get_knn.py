import time
import json
import random
import argparse
import logging
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

import vision_transformer as vits

from project_utils.infomap_cluster_utils import cluster_by_semi_infomap, get_dist_nbr


# ============================================
# Utility Functions
# ============================================

def init_seed_torch(seed=1):
    """Initialize all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logger(name='get_knn', log_file=None):
    """Setup logger with console and optional file output"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy for labeled data subset
    Replaces -1 with unique cluster IDs
    Returns both raw accuracy and error-adjusted accuracy
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.copy()
    assert y_pred.size == y_true.size

    # Count valid clusters (excluding -1)
    valid_clusters = np.unique(y_pred[y_pred != -1])
    num_valid_clusters = len(valid_clusters)
    
    # Calculate cluster count estimation error rate
    num_real_clusters = len(np.unique(y_true))
    cluster_error_rate = abs(num_real_clusters - num_valid_clusters) / num_valid_clusters if num_valid_clusters > 0 else 1.0
    
    # Handle -1: replace each -1 with a different cluster ID
    max_pred = np.max(y_pred[y_pred != -1]) if np.any(y_pred != -1) else -1
    new_cluster_id = max_pred + 1
    for i in range(len(y_pred)):
        if y_pred[i] == -1:
            y_pred[i] = new_cluster_id
            new_cluster_id += 1
    
    # Hungarian algorithm for optimal assignment
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    ind = linear_sum_assignment(w.max() - w)
    ind = np.vstack(ind).T
    
    total_acc = sum([w[i, j] for i, j in ind])
    total_acc /= y_pred.size
    
    return total_acc, total_acc * (1 - cluster_error_rate), num_valid_clusters, cluster_error_rate


# ============================================
# Token Cache Functions
# ============================================

def to_torch(ndarray):
    """Convert numpy array to torch tensor"""
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError(f"Cannot convert {type(ndarray)} to torch tensor")
    return ndarray


def _cache_paths(cache_dir: Path):
    """Get cache file paths"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "meta": cache_dir / "meta.json",
        "tokens": cache_dir / "tokens_fp16.dat",
        "targets": cache_dir / "targets.npy",
        "ifl": cache_dir / "if_labeled.npy",
    }


def _has_cache(cache_dir: Path) -> bool:
    """Check if cache exists"""
    p = _cache_paths(cache_dir)
    return all(path.exists() for path in p.values())


@torch.no_grad()
def _build_token_cache(cluster_loader, model, args):
    """Build token cache for efficient feature extraction"""
    device = args.device
    cache_dir = Path(args.token_cache_dir)
    p = _cache_paths(cache_dir)

    start_t = time.time()
    model.eval()
    N = len(cluster_loader.dataset)
    write_ptr, first = 0, True

    for _item in cluster_loader:
        imgs = to_torch(_item[0]).to(device, non_blocking=True)
        tokens = model.forward_until(imgs, end_block=args.grad_from_block)
        B, S, D = tokens.shape

        if first:
            mm = np.memmap(p["tokens"], mode="w+", dtype=np.float16, shape=(N, S, D))
            meta = {"N": int(N), "seq_len": int(S), "embed_dim": int(D),
                    "start_block": int(args.grad_from_block), "dtype": "float16"}
            with open(p["meta"], "w") as f:
                json.dump(meta, f)
            targets_buf = np.empty((N,), dtype=np.int64)
            ifl_buf = np.empty((N,), dtype=np.bool_)
            first = False

        mm[write_ptr: write_ptr+B] = tokens.detach().cpu().to(torch.float16).numpy()
        targets_buf[write_ptr: write_ptr+B] = _item[1].cpu().numpy()
        ifl_buf[write_ptr: write_ptr+B] = _item[3][:,0].bool().cpu().numpy()
        write_ptr += B

    del mm
    np.save(p["targets"], targets_buf)
    np.save(p["ifl"], ifl_buf)
    elapsed = time.time() - start_t
    args.logger.info(f"[TokenCache] Built at {str(cache_dir)} | N={N} | took {elapsed:.2f}s")
    return elapsed


def _load_token_cache(cache_dir: Path):
    """Load token cache from disk"""
    p = _cache_paths(cache_dir)
    with open(p["meta"], "r") as f:
        meta = json.load(f)
    N, S, D = meta["N"], meta["seq_len"], meta["embed_dim"]
    tokens_mm = np.memmap(p["tokens"], mode="r", dtype=np.float16, shape=(N, S, D))
    targets = np.load(p["targets"])
    ifl = np.load(p["ifl"])
    return tokens_mm, targets, ifl, meta


@torch.no_grad()
def _extract_from_cache(model, cache_dir, args):
    """Extract features from cached tokens"""
    device = args.device
    bs = args.cls_export_bs

    tokens_mm, targets_np, ifl_np, meta = _load_token_cache(Path(cache_dir))
    N = meta["N"]
    feats_buckets = []
    labels_buckets, ifl_buckets = [], []

    t0 = time.time()
    for i in range(0, N, bs):
        j = min(i+bs, N)
        chunk_np = np.array(tokens_mm[i:j], copy=True)
        chunk_t = torch.from_numpy(chunk_np)
        chunk_t = chunk_t.pin_memory()
        chunk = chunk_t.to(device=device, dtype=torch.float32, non_blocking=True)
        cls = model.forward_from_tokens(chunk, start_block=args.grad_from_block, return_all=False)

        feats_buckets.append(cls.detach().cpu())
        labels_buckets.append(torch.from_numpy(targets_np[i:j]))
        ifl_buckets.append(torch.from_numpy(ifl_np[i:j]))

        del chunk, cls
    
    t_feat = time.time() - t0

    results = {
        "features": torch.cat(feats_buckets, 0),
        "labels": torch.cat(labels_buckets, 0),
        "if_labeled": torch.cat(ifl_buckets, 0)
    }

    args.logger.info(f"[TokenCache] Export CLS with cache took {t_feat:.2f}s")
    return results, t_feat


# ============================================
# K Selection Functions
# ============================================

def find_optimal_k(labels_labelled, feat_nbrs, feat_dists, args):
    """
    Find optimal K using coarse-to-fine search strategy
    1. Coarse search: test on logarithmic scale
    2. Fine search: detailed search around best coarse K
       - If range > 100: search with step size of 10
       - Otherwise: search all values
    """
    logger = args.logger
    len_of_labeled = len(labels_labelled)
    
    # Coarse search on logarithmic scale
    # coarse_k_values = [512, 256, 128, 64, 32, 16, 8, 4]
    coarse_k_values = [k for k in [512, 256, 128, 64, 32, 16, 8, 4] if k <= args.len_cluster_dataset]
    coarse_results = []
    
    logger.info("=" * 60)
    logger.info("Starting coarse K selection...")
    logger.info("=" * 60)
    
    for Kmax in coarse_k_values:
        pseudo_labels = cluster_by_semi_infomap(
            feat_nbrs[:, :Kmax], 
            feat_dists[:, :Kmax],
            min_sim=args.minsim,
            cluster_num=args.k2,
            label_mark=None,
            if_labeled=None,
            args=args
        ).astype(np.intp)
        
        acc, err_acc, est_k, err_rate = cluster_acc(labels_labelled, pseudo_labels[:len_of_labeled])
        
        coarse_results.append({
            'k': Kmax,
            'acc': acc,
            'err_acc': err_acc,
            'est_k': est_k,
            'err_rate': err_rate
        })
        
        logger.info(f"K={Kmax:3d}: ACC={acc:.4f}, ERR_ACC={err_acc:.4f}, "
                   f"Est_K={est_k:3d}, Err_Rate={err_rate:.4f}")
    
    # Find best coarse K based on ACC
    best_coarse = max(coarse_results, key=lambda x: x['acc'])
    best_coarse_k = best_coarse['k']
    best_coarse_idx = coarse_k_values.index(best_coarse_k)
    
    logger.info(f"\nBest coarse K={best_coarse_k} with ACC={best_coarse['acc']:.4f}")
    
    # Define fine search range
    if best_coarse_idx == 0:  # K=512 is best
        fine_k_start = coarse_k_values[best_coarse_idx + 1] + 1
        fine_k_end = coarse_k_values[0]
    elif best_coarse_idx == len(coarse_k_values) - 1:  # K=4 is best
        fine_k_start = coarse_k_values[-1]
        fine_k_end = coarse_k_values[best_coarse_idx - 1] - 1
    else:
        # Search between previous and next coarse values
        fine_k_start = coarse_k_values[best_coarse_idx + 1] + 1
        fine_k_end = coarse_k_values[best_coarse_idx - 1] - 1
    
    # Determine search strategy based on range size
    range_size = fine_k_end - fine_k_start + 1
    
    # Fine search
    logger.info("=" * 60)
    logger.info(f"Starting fine K selection in range [{fine_k_start}, {fine_k_end}]...")
    
    if range_size > 100:
        # For large ranges, use step size of 10
        logger.info(f"Range size {range_size} > 100, using step size of 10")
        
        # Create K values list with step 10, plus endpoints
        fine_k_values = [fine_k_start]  # Start point
        
        # Add multiples of 10 in the range
        current = ((fine_k_start // 10) + 1) * 10
        while current < fine_k_end:
            fine_k_values.append(current)
            current += 10
        
        # Add endpoint if not already included
        if fine_k_values[-1] != fine_k_end:
            fine_k_values.append(fine_k_end)
        
        logger.info(f"Testing K values: {fine_k_values}")
    else:
        # For smaller ranges, test all values
        logger.info(f"Range size {range_size} <= 100, testing all values")
        fine_k_values = list(range(fine_k_start, fine_k_end + 1))
    
    logger.info("=" * 60)
    
    fine_results = []
    
    for i, Kmax in enumerate(fine_k_values):
        pseudo_labels = cluster_by_semi_infomap(
            feat_nbrs[:, :Kmax], 
            feat_dists[:, :Kmax],
            min_sim=args.minsim,
            cluster_num=args.k2,
            label_mark=None,
            if_labeled=None,
            args=args
        ).astype(np.intp)
        
        acc, err_acc, est_k, err_rate = cluster_acc(labels_labelled, pseudo_labels[:len_of_labeled])
        
        fine_results.append({
            'k': Kmax,
            'acc': acc,
            'err_acc': err_acc,
            'est_k': est_k,
            'err_rate': err_rate
        })
        
        logger.info(f"K={Kmax:3d}: ACC={acc:.4f}, ERR_ACC={err_acc:.4f}, "
                    f"Est_K={est_k:3d}, Err_Rate={err_rate:.4f}")
    
    # Find best fine K based on ERR_ACC
    best_fine = max(fine_results, key=lambda x: x['err_acc'])
    
    logger.info("=" * 60)
    logger.info(f"Optimal K={best_fine['k']} selected with:")
    logger.info(f"  - ACC: {best_fine['acc']:.4f}")
    logger.info(f"  - ERR_ACC: {best_fine['err_acc']:.4f}")
    logger.info(f"  - Estimated clusters: {best_fine['est_k']}")
    logger.info(f"  - Error rate: {best_fine['err_rate']:.4f}")
    logger.info("=" * 60)
    
    return best_fine['k'], best_fine

# def find_optimal_k(labels_labelled, feat_nbrs, feat_dists, args):
    """
    Find optimal K using coarse-to-fine search strategy
    1. Coarse search: test on logarithmic scale
    2. Fine search: detailed search around best coarse K
       - If best_k > 500 and range > 100: search with step size of 50
       - Elif range > 100: search with step size of 10
       - Otherwise: search all values
    """
    logger = args.logger
    len_of_labeled = len(labels_labelled)
    
    # Coarse search on logarithmic scale (增加1024)
    coarse_k_values = [1024, 512, 256, 128, 64, 32, 16, 8, 4]
    coarse_results = []
    
    logger.info("=" * 60)
    logger.info("Starting coarse K selection...")
    logger.info("=" * 60)
    
    for Kmax in coarse_k_values:
        pseudo_labels = cluster_by_semi_infomap(
            feat_nbrs[:, :Kmax], 
            feat_dists[:, :Kmax],
            min_sim=args.minsim,
            cluster_num=args.k2,
            label_mark=None,
            if_labeled=None,
            args=args
        ).astype(np.intp)
        
        acc, err_acc, est_k, err_rate = cluster_acc(labels_labelled, pseudo_labels[:len_of_labeled])
        
        coarse_results.append({
            'k': Kmax,
            'acc': acc,
            'err_acc': err_acc,
            'est_k': est_k,
            'err_rate': err_rate
        })
        
        logger.info(f"K={Kmax:3d}: ACC={acc:.4f}, ERR_ACC={err_acc:.4f}, "
                   f"Est_K={est_k:3d}, Err_Rate={err_rate:.4f}")
    
    # Find best coarse K based on ACC
    best_coarse = max(coarse_results, key=lambda x: x['acc'])
    best_coarse_k = best_coarse['k']
    best_coarse_idx = coarse_k_values.index(best_coarse_k)
    
    logger.info(f"\nBest coarse K={best_coarse_k} with ACC={best_coarse['acc']:.4f}")
    
    # Define fine search range
    if best_coarse_idx == 0:  # K=1024 is best
        fine_k_start = 513
        fine_k_end = 1024
    elif best_coarse_idx == len(coarse_k_values) - 1:  # K=4 is best
        fine_k_start = 4
        fine_k_end = 7
    else:
        # Search between previous and next coarse values
        fine_k_start = coarse_k_values[best_coarse_idx + 1] + 1
        fine_k_end = coarse_k_values[best_coarse_idx - 1] - 1
    
    # Determine search strategy based on range size
    range_size = fine_k_end - fine_k_start + 1
    
    # Fine search
    logger.info("=" * 60)
    logger.info(f"Starting fine K selection in range [{fine_k_start}, {fine_k_end}]...")
    
    if range_size > 100:
        # 根据best_coarse_k决定步长：>500时用50，否则用10
        if best_coarse_k > 500:
            step_size = 50
            logger.info(f"Range size {range_size} > 100 and best_coarse_k > 500, using step size of 50")
        else:
            step_size = 10
            logger.info(f"Range size {range_size} > 100, using step size of 10")
        
        # Create K values list with determined step size, plus endpoints
        fine_k_values = [fine_k_start]  # Start point
        
        # Add multiples of step_size in the range
        current = ((fine_k_start // step_size) + 1) * step_size
        while current < fine_k_end:
            fine_k_values.append(current)
            current += step_size
        
        # Add endpoint if not already included
        if fine_k_values[-1] != fine_k_end:
            fine_k_values.append(fine_k_end)
        
        logger.info(f"Testing K values: {fine_k_values}")
    else:
        # For smaller ranges, test all values
        logger.info(f"Range size {range_size} <= 100, testing all values")
        fine_k_values = list(range(fine_k_start, fine_k_end + 1))
    
    logger.info("=" * 60)
    
    fine_results = []
    
    for i, Kmax in enumerate(fine_k_values):
        pseudo_labels = cluster_by_semi_infomap(
            feat_nbrs[:, :Kmax], 
            feat_dists[:, :Kmax],
            min_sim=args.minsim,
            cluster_num=args.k2,
            label_mark=None,
            if_labeled=None,
            args=args
        ).astype(np.intp)
        
        acc, err_acc, est_k, err_rate = cluster_acc(labels_labelled, pseudo_labels[:len_of_labeled])
        
        fine_results.append({
            'k': Kmax,
            'acc': acc,
            'err_acc': err_acc,
            'est_k': est_k,
            'err_rate': err_rate
        })
        
        logger.info(f"K={Kmax:3d}: ACC={acc:.4f}, ERR_ACC={err_acc:.4f}, "
                    f"Est_K={est_k:3d}, Err_Rate={err_rate:.4f}")
    
    # Find best fine K based on ERR_ACC
    best_fine = max(fine_results, key=lambda x: x['err_acc'])
    
    logger.info("=" * 60)
    logger.info(f"Optimal K={best_fine['k']} selected with:")
    logger.info(f"  - ACC: {best_fine['acc']:.4f}")
    logger.info(f"  - ERR_ACC: {best_fine['err_acc']:.4f}")
    logger.info(f"  - Estimated clusters: {best_fine['est_k']}")
    logger.info(f"  - Error rate: {best_fine['err_rate']:.4f}")
    logger.info("=" * 60)
    
    return best_fine['k'], best_fine
# ============================================
# Argument Parser
# ============================================

def get_args():
    parser = argparse.ArgumentParser(description='MGCE Training with Adaptive K Selection')
    
    # Basic settings
    parser.add_argument('--cluster_batch_size', type=int, default=1024,
                        help='Batch size for clustering')
    parser.add_argument('--cluster_num_workers', type=int, default=2,
                        help='Number of workers for cluster dataloader')
    
    # Dataset settings
    parser.add_argument('--dataset_name', type=str, default='aircraft',
                        choices=['cifar100', 'cub', 'aircraft', 'scars', 'herbarium_19', 'imagenet_100','Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Chromista', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia', 'nabirds'],
                        help='Dataset to use')
    parser.add_argument('--prop_train_labels', type=float, default=0.5,
                        help='Proportion of labeled training data')
    parser.add_argument('--use_ssb_splits', action='store_true', default=True,
                        help='Use SSB splits')
    
    # Training settings
    parser.add_argument('--grad_from_block', type=int, default=11,
                        help='Fine-tune from this transformer block')
    
    # Transform settings
    parser.add_argument('--transform', type=str, default='imagenet',
                        help='Type of data augmentation')
    parser.add_argument('--n_views', type=int, default=2,
                        help='Number of augmented views')
    
    # Clustering settings
    parser.add_argument('--k2', type=int, default=4,
                        help='Number of clusters for infomap')
    parser.add_argument('--use_hard', action='store_true', default=False,
                        help='Use hard pseudo labels')
    parser.add_argument('--num_instances', type=int, default=16,
                        help='Number of instances per class')
    parser.add_argument('--max_sim', action='store_true', default=True,
                        help='Use maximum similarity')
    parser.add_argument('--minsim', type=float, default=0.6,
                        help='Minimum similarity threshold')
    
    # Path settings
    parser.add_argument('--cub_root', type=str, default='')
    # parser.add_argument('--cifar_10_root', type=str, default=cifar_10_root)
    parser.add_argument('--cifar_100_root', type=str, default='')
    parser.add_argument('--cars_root', type=str, default='')
    parser.add_argument('--pets_root', type=str, default='')
    parser.add_argument('--aircraft_root', type=str, default='')
    parser.add_argument('--imagenet_root', type=str, default='')
    parser.add_argument('--herbarium_dataroot', type=str, default='')
    parser.add_argument('--osr_split_dir', type=str, default='')
    parser.add_argument('--dino_pretrain_path', type=str, default='')
    parser.add_argument('--inaturalist_root', type=str, default='')
    parser.add_argument('--nabirds_root', type=str, default='')

    
    # Experiment settings
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--use_gpu_knn', action='store_true', default=True,
                        help='Use GPU for KNN computation')
    parser.add_argument('--use_token_cache', action='store_true', default=True,
                        help='Use token caching for efficiency')
    parser.add_argument('--token_cache_dir', type=str,
                        default='',
                        help='Directory for token cache')
    parser.add_argument('--cls_export_bs', type=int, default=1024,
                        help='Batch size for CLS export')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='',
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--exp_name', type=str, default='Get_KNN',
                        help='Experiment name (auto-generated if not provided)')
    
    parser.add_argument("--cifar100_sample_subset", action="store_true", help="use cineca gpu")# 默认是 False

    return parser.parse_args()


# ============================================
# Model Setup
# ============================================

def setup_model(args):
    """Initialize and configure the model"""
    model = vits.__dict__['vit_base']()
    state_dict = torch.load(args.dino_pretrain_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Freeze all parameters initially
    for m in model.parameters():
        m.requires_grad = False
    
    return model


# ============================================
# Main Training Function
# ============================================

def main():
    # Parse arguments
    args = get_args()
    
    # Setup device
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    if args.exp_name is None:
        args.exp_name = f"{args.dataset_name}_getknn_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    args.output_dir = Path(args.output_dir) / args.exp_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    log_file = args.output_dir / 'get_knn.log'
    args.logger = setup_logger('MGCE', log_file=str(log_file))
    
    # Initialize random seeds
    init_seed_torch(args.seed)
    
    # Log configuration
    args.logger.info("=" * 60)
    args.logger.info("MGCE Training with Adaptive K Selection")
    args.logger.info("=" * 60)
    args.logger.info(f"Configuration:")
    for key, value in vars(args).items():
        if key != 'logger':
            args.logger.info(f"  {key}: {value}")
    args.logger.info("=" * 60)
    
    
    # Get class splits
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    
    args.logger.info(f"Number of labeled classes: {args.num_labeled_classes}")
    args.logger.info(f"Number of unlabeled classes: {args.num_unlabeled_classes}")
    args.logger.info(f"Train classes: {args.train_classes}")
    args.logger.info(f"Unlabeled classes: {args.unlabeled_classes}")
    
    # Setup model parameters
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224

    
    # Initialize model
    model = setup_model(args)
    model = model.to(args.device)
    
    # Get datasets
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    _, cluster_dataset = get_datasets(args.dataset_name, train_transform, test_transform, args)
    args.len_cluster_dataset = len(cluster_dataset)
    
    args.logger.info(f"Cluster dataset size: {len(cluster_dataset)}")
    args.logger.info(f"Labelled dataset size: {len(cluster_dataset.labelled_dataset)}")
    args.logger.info(f"Unlabelled dataset size: {len(cluster_dataset.unlabelled_dataset)}")
    
    # Create cluster dataloader
    cluster_loader = DataLoader(
        deepcopy(cluster_dataset),
        num_workers=args.cluster_num_workers,
        batch_size=args.cluster_batch_size,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=6,
        drop_last=False,
        pin_memory_device="cuda:0",
    )
    
    # Handle token cache
    use_cache = args.use_token_cache
    cache_dir = Path(args.token_cache_dir)
    
    if use_cache and not _has_cache(cache_dir):
        args.logger.info("Building token cache...")
        _build_token_cache(cluster_loader, model, args)
    
    # Extract features
    args.logger.info("Extracting features...")
    results_dict, t_cache_feat = _extract_from_cache(model, cache_dir, args)
    
    features = results_dict["features"]
    features_array = F.normalize(features, dim=1).cpu().numpy()
    
    # Compute K-nearest neighbors
    args.logger.info("Computing K-nearest neighbors...")
    feat_dists, feat_nbrs = get_dist_nbr(
        features=features_array,
        k=512,  # Maximum K for coarse search
        knn_method='faiss-gpu' if args.use_gpu_knn else 'faiss',
        device=0 if args.use_gpu_knn else None
    )
    
    # Prepare labels for evaluation
    len_of_labeled = len(cluster_dataset.labelled_dataset)
    labels = results_dict["labels"].numpy()
    labels_labelled = labels[:len_of_labeled]
    

    args.logger.info("Starting automatic K selection...")
    optimal_k, k_stats = find_optimal_k( 
        labels_labelled,
        feat_nbrs, 
        feat_dists, 
        args
    )
    
    # Save K selection results
    k_results_file = args.output_dir / 'k_selection_results.json'
    with open(k_results_file, 'w') as f:
        json.dump(k_stats, f, indent=2)
    args.logger.info(f"K selection results saved to {k_results_file}")
    args.logger.info(f"Final KNN: {optimal_k}")
        

if __name__ == "__main__":
    main()