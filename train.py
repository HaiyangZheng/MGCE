import os
import math
import time
import json
import random
import argparse

from copy import deepcopy
from pathlib import Path

import wandb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds

from model import (
    BranchHead, 
    BranchHead_New, 
    info_nce_logits, 
    SupConLoss, 
    DistillLoss, 
    ContrastiveLearningViewGenerator,
    get_params_groups
)
import vision_transformer as vits

from project_utils.infomap_cluster_utils import cluster_by_semi_infomap, get_dist_nbr, generate_cluster_features
from project_utils.cluster_memory_utils import ClusterMemory as DCCL_ClusterMemory
from project_utils.data_utils import IterLoader, FakeLabelDataset_3head
#-------------------------------------seed----------------------------------------#
def init_seed_torch(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#-------------------------------------Inference----------------------------------------#

def test_with_pseudo_label_full(name, pred, targets, save_name='Train ACC Unlabelled', epoch=0):

    if len(pred) != len(targets):
        raise ValueError("Length of predictions and targets must be the same.")
    
    args.logger.info(f"Length of pred: {len(pred)}")
    
    mask = [target < len(args.train_classes) for target in targets]
    mask = np.array(mask).astype(bool)

    pred_2 = np.copy(pred)
    max_pred = np.max(pred_2)
    new_class_label = max_pred + 1
    for i in range(len(pred_2)):
        if pred_2[i] == -1:
            pred_2[i] = new_class_label
            new_class_label += 1
    
    all_acc_2, old_acc_2, new_acc_2 = log_accs_from_preds(
        y_true=targets, y_pred=pred_2, mask=mask, T=epoch,
        eval_funcs=args.eval_funcs, save_name=save_name, args=args)
    
    args.logger.info(f"{name} - All Acc: {all_acc_2}, Old Acc: {old_acc_2}, New Acc: {new_acc_2}")
    return all_acc_2, old_acc_2, new_acc_2


#-------------------------------------SimGCD Loss----------------------------------------#
def simgcd_loss(studdent_out_put, mask_lab, class_labels, criterion, epoch):
    student_proj, student_out = studdent_out_put[0], studdent_out_put[1]
    teacher_out = student_out.detach()

    # clustering, sup
    sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

    # clustering, unsup
    cluster_loss = criterion(student_out, teacher_out, epoch)
    avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
    me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
    cluster_loss += args.memax_weight * me_max_loss

    # represent learning, unsup
    contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)  # include l2
    contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

    # representation learning, sup
    student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
    student_proj = torch.nn.functional.normalize(student_proj, dim=-1)  # l2
    sup_con_labels = class_labels[mask_lab]
    sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)  # does not include l2

    return cls_loss, cluster_loss, contrastive_loss, sup_con_loss


#-------------------------------------Align Loss----------------------------------------#
def align_to_base_soft_js(dccl_out, dccl_centers, base_centers, base_output):

    sim_to_base = dccl_centers @ base_centers.t()

    norm_features = F.normalize(dccl_out, p=2, dim=1)

    similarity = norm_features @ dccl_centers.t()

    outputs = similarity @ sim_to_base
    outputs = F.softmax(outputs, dim=1)  
    pseudo_label = F.softmax(base_output, dim=1)  

    log_prob_outputs = F.log_softmax(outputs, dim=1)
    log_prob_labels = F.log_softmax(pseudo_label, dim=1)

    kl_div1 = F.kl_div(log_prob_outputs, pseudo_label, reduction='batchmean')
    kl_div2 = F.kl_div(log_prob_labels, outputs, reduction='batchmean')

    js_divergence = 0.5 * (kl_div1 + kl_div2)

    return js_divergence


#-------------------------------------utils----------------------------------------#
def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

@torch.no_grad()
def extract_features_with_headlist(backbone, project_dict, data_loader, args):
    backbone.eval()
    
    dccl_names = ['DCCL', 'DCCL2', 'DCCL3']
    dccl_projectors = [project_dict[name] for name in dccl_names]
    
    for projector in dccl_projectors:
        projector.eval()
    
    all_features = [[] for _ in range(3)]  
    all_labels = []
    all_if_labeled = []
    
    for _item in data_loader:
        imgs = to_torch(_item[0]).to(args.device)
        targets = _item[1]
        if_train = _item[3][:, 0].bool()
        
        cls_token = backbone(imgs, False)
        
        proj_out_0 = dccl_projectors[0](cls_token)
        proj_out_1 = dccl_projectors[1](cls_token)
        proj_out_2 = dccl_projectors[2](cls_token)
        
        all_features[0].append(proj_out_0[args.feature_output_index].data.cpu())
        all_features[1].append(proj_out_1[args.feature_output_index].data.cpu())
        all_features[2].append(proj_out_2[args.feature_output_index].data.cpu())
        
        all_labels.append(targets)
        all_if_labeled.append(if_train)
    
    results = {}
    for i, name in enumerate(dccl_names):
        results[name + "_features"] = torch.cat(all_features[i], dim=0)
        results[name + "_labels"] = torch.cat(all_labels, dim=0)
        results[name + "_if_labeled"] = torch.cat(all_if_labeled, dim=0)

    return results

def _cache_paths(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "meta": cache_dir / "meta.json",
        "tokens": cache_dir / "tokens_fp16.dat",
        "targets": cache_dir / "targets.npy",
        "ifl": cache_dir / "if_labeled.npy",
    }

def _has_cache(cache_dir: Path) -> bool:
    p = _cache_paths(cache_dir)
    return p["meta"].exists() and p["tokens"].exists() and p["targets"].exists() and p["ifl"].exists()

@torch.no_grad()
def _build_token_cache(cluster_loader, model, args):

    device = args.device
    cache_dir = Path(getattr(args, "token_cache_dir", "token_cache"))
    p = _cache_paths(cache_dir)

    start_t = time.time()
    model.eval()
    N = len(cluster_loader.dataset)
    write_ptr, first = 0, True

    for _item in cluster_loader:
        imgs = to_torch(_item[0]).to(device, non_blocking=True)
        tokens = model.forward_until(imgs, end_block=args.grad_from_block)  # [B,S,D]
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
    print(f"[TokenCache] Built at {str(cache_dir)} | N={N} | took {elapsed:.2f}s")
    return elapsed  

def _load_token_cache(cache_dir: Path):
    p = _cache_paths(cache_dir)
    with open(p["meta"], "r") as f:
        meta = json.load(f)
    N, S, D = meta["N"], meta["seq_len"], meta["embed_dim"]
    tokens_mm = np.memmap(p["tokens"], mode="r", dtype=np.float16, shape=(N, S, D))
    targets = np.load(p["targets"])
    ifl = np.load(p["ifl"])
    return tokens_mm, targets, ifl, meta

@torch.no_grad()
def _extract_from_cache(model, projector_dict, cache_dir, args):
    device = args.device
    bs = int(getattr(args, "cls_export_bs", 4096))
    names = args.dccl_heads
    projs = [projector_dict[n] for n in names]

    tokens_mm, targets_np, ifl_np, meta = _load_token_cache(Path(cache_dir))
    N = meta["N"]
    feats_buckets = [[] for _ in range(len(names))]
    labels_buckets, ifl_buckets = [], []

    cls_buckets = []

    t0 = time.time()
    for i in range(0, N, bs):
        j = min(i+bs, N)
        # chunk = torch.from_numpy(tokens_mm[i:j]).to(device=device, dtype=torch.float32, non_blocking=True)  # [B,S,D]
        chunk_np = np.array(tokens_mm[i:j], copy=True)
        chunk_t  = torch.from_numpy(chunk_np)
        chunk_t = chunk_t.pin_memory()
        chunk = chunk_t.to(device=device, dtype=torch.float32, non_blocking=True) 
        cls = model.forward_from_tokens(chunk, start_block=args.grad_from_block, return_all=False)  # [B,D]

        for k, proj in enumerate(projs):
            out = proj(cls)
            feats_buckets[k].append(out[args.feature_output_index].detach().cpu())

        labels_buckets.append(torch.from_numpy(targets_np[i:j]))
        ifl_buckets.append(torch.from_numpy(ifl_np[i:j]))
        cls_buckets.append(cls.detach().cpu())

        del chunk, cls, out
    t_feat = time.time() - t0

    results = {}
    labels_cat = torch.cat(labels_buckets, 0)
    ifl_cat = torch.cat(ifl_buckets, 0)
    cls_cat = torch.cat(cls_buckets, 0)
    for k, name in enumerate(names):
        results[name+"_features"] = torch.cat(feats_buckets[k], 0)
        results[name+"_labels"] = labels_cat
        results[name+"_if_labeled"] = ifl_cat
    results['cls'] = cls_cat

    print(f"[TokenCache] Export CLS+projectors with cache took {t_feat:.2f}s")
    return results, t_feat

@torch.no_grad()
def DCCL(cluster_dataset, cluster_loader, model, projector_dict, epoch, args):

    print('==> Create pseudo labels for unlabeled data')

    use_cache = bool(getattr(args, "use_token_cache", True))
    cache_dir = Path(getattr(args, "token_cache_dir", "token_cache"))

    if use_cache and epoch == 0 and not _has_cache(cache_dir):
        print("==> Building token cache (epoch=0)")
        _build_token_cache(cluster_loader, model, args)

    if use_cache and _has_cache(cache_dir):
        results_dict, _ = _extract_from_cache(model, projector_dict, cache_dir, args)
    else:
        results_dict = extract_features_with_headlist(model, projector_dict, cluster_loader, args)

    # ========= (2)(3)(4)(5) 三分支阶段计时 =========
    dccl_names = args.dccl_heads

    memory_dict = {}
    pseudo_labels_dict = {}

    len_of_labeled = len(cluster_dataset.labelled_dataset)
    

    # test backbone
    cls = results_dict['cls']
    label_mark = results_dict["DCCL_labels"]
    if_labeled = results_dict["DCCL_if_labeled"]

    item_targets = label_mark[len_of_labeled:].cpu().numpy()

    features_array = F.normalize(cls, dim=1).cpu().numpy()
    feat_dists, feat_nbrs = get_dist_nbr(
        features=features_array,
        k=args.k1_dict['DCCL'],
        knn_method='faiss-gpu',
        device=0
    )

    pseudo_labels, _  = cluster_by_semi_infomap(
        feat_nbrs, feat_dists,
        min_sim=args.eps_dict['DCCL'],
        cluster_num=args.k2,
        label_mark=label_mark,
        if_labeled=if_labeled,
        args=args,
        head='cls',
        if_combine=True
    )

    all_acc, old_acc, new_acc = test_with_pseudo_label_full(
        "MGCE Inference",
        pred=pseudo_labels[len_of_labeled:],
        targets=item_targets
    )

    wandb.log({
        f"Backbone_All_ACC": all_acc,
        f"Backbone_Old_ACC": old_acc,
        f"Backbone_New_ACC": new_acc,
        "epoch": epoch 
    })

    for name in dccl_names:

        features = results_dict[name + "_features"]
        label_mark = results_dict[name + "_labels"]
        if_labeled = results_dict[name + "_if_labeled"]

        features_array = F.normalize(features, dim=1).cpu().numpy()

        pseudo_labels = cluster_by_semi_infomap(
            feat_nbrs, feat_dists,
            min_sim=args.eps_dict[name],
            cluster_num=args.k2,
            label_mark=label_mark,
            if_labeled=if_labeled,
            args=args,
            head=name
        ).astype(np.intp)



        pseudo_labels_dict[name] = pseudo_labels

        num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        
        cluster_features = generate_cluster_features(pseudo_labels, features)

        memory = DCCL_ClusterMemory(
            num_features=args.feat_dim, num_samples=num_cluster, temp=args.temp,
            momentum=args.memory_momentum, use_hard=args.use_hard
        ).to(args.device)
        memory.features = F.normalize(cluster_features, dim=1).to(args.device)
        if getattr(args, "fp16", True):
            memory.features = memory.features.half()
        memory_dict[name] = memory

        del features_array

    pseudo_labeled_dataset, item_targets_list = [], []
    for i, (_item, labels_dccl, labels_dccl2, labels_dccl3, _if_labeled) in enumerate(zip(cluster_dataset.data, pseudo_labels_dict['DCCL'], pseudo_labels_dict['DCCL2'], pseudo_labels_dict['DCCL3'], if_labeled)):
        item_targets_list.append(cluster_dataset.targets[i])

        if labels_dccl != -1 and labels_dccl2 != -1 and labels_dccl3 != -1:
            if isinstance(_item, str):
                pseudo_labeled_dataset.append((_item, labels_dccl.item(), labels_dccl2.item(), labels_dccl3.item(), cluster_dataset.targets[i], _if_labeled.item()))
            elif args.dataset_name in ['imagenet_100', 'herbarium_19']:
                pseudo_labeled_dataset.append((_item[0], labels_dccl.item(), labels_dccl2.item(), labels_dccl3.item(), cluster_dataset.targets[i], _if_labeled.item()))
            else:
                pseudo_labeled_dataset.append((_item[1], labels_dccl.item(), labels_dccl2.item(), labels_dccl3.item(), cluster_dataset.targets[i], _if_labeled.item()))

    dccl_dataset = FakeLabelDataset_3head(pseudo_labeled_dataset, root=None, transform=train_transform)


    contrastive_cluster_train_loader = IterLoader(
        DataLoader(
            dccl_dataset,
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=0,
            # pin_memory=True,
            shuffle=True,
            # sampler=infinite_sampler
        )
    )
    contrastive_cluster_train_loader.new_epoch()

    return contrastive_cluster_train_loader, memory_dict


#-------------------------------------Train----------------------------------------#

def train(model, projector_dict, train_loader, cluster_loader, args):
    model.to(args.device)
    projector_dict.to(args.device)

    params_groups = get_params_groups(model)
    for projector in projector_dict.values():
        params_groups += get_params_groups(projector)

    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )

    DCCL_contrastive_cluster_weight_schedule = np.concatenate((
        np.linspace(args.DCCL_contrastive_cluster_weight * 0.1,
                    args.DCCL_contrastive_cluster_weight, args.contrastive_cluster_epochs),
        np.ones(args.epochs - args.contrastive_cluster_epochs) * args.DCCL_contrastive_cluster_weight
    ))

    DCCL_align_weight_schedule = np.concatenate((
        np.linspace(args.DCCL_align_weight * 0.1,
                    args.DCCL_align_weight, args.contrastive_cluster_epochs),
        np.ones(args.epochs - args.contrastive_cluster_epochs) * args.DCCL_align_weight
    ))


    # ========== 开始训练循环 ========== #
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
    
        if args.use_contrastive_cluster and epoch % args.dccl_update_freq == 0:
            model.eval()
            projector_dict.eval()
            dccl_loader, memory_dict = DCCL(
                cluster_dataset=cluster_dataset,
                cluster_loader=cluster_loader,
                model=model, 
                projector_dict=projector_dict, 
                epoch=epoch, 
                args=args
            )
        
        model.train()
        projector_dict.train()

        for batch_idx, batch in enumerate(train_loader):
            
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            class_labels = class_labels.cuda(non_blocking=True)
            mask_lab = mask_lab.cuda(non_blocking=True).bool()
            main_imgs = torch.cat(images, dim=0).cuda(non_blocking=True)  # [B*n_views, C, H, W]

            dccl_heads = args.dccl_heads
            dccl_labels_head = {}
            
            imgs, labels_dccl, labels_dccl2, labels_dccl3, idxs = dccl_loader.next()
            if isinstance(imgs, (list, tuple)):
                imgs = torch.cat(imgs, dim=0)
                labels_dccl_copy = labels_dccl.detach().clone()
                labels_dccl = torch.cat((labels_dccl, labels_dccl_copy), dim=0)
                labels_dccl2_copy = labels_dccl2.detach().clone()
                labels_dccl2 = torch.cat((labels_dccl2, labels_dccl2_copy), dim=0)
                labels_dccl3_copy = labels_dccl3.detach().clone()
                labels_dccl3 = torch.cat((labels_dccl3, labels_dccl3_copy), dim=0)
            dccl_images = imgs.to(args.device)

            for name in dccl_heads:
                dccl_labels_head[name] = labels_dccl if name == 'DCCL' else (
                    labels_dccl2 if name == 'DCCL2' else labels_dccl3
                )

            all_images = [main_imgs]
            all_images.append(dccl_images) 

            cum_sizes = main_imgs.shape[0]  # [0, size1, size1+size2, ...]
            
            big_batch = torch.cat(all_images, dim=0)  # [total_batch_size, C, H, W]


            with torch.cuda.amp.autocast(fp16_scaler is not None):
                all_cls_tokens = model(big_batch, return_all_patches=False)  # [total_batch, dim]

            
            with torch.cuda.amp.autocast(fp16_scaler is not None):

                total_loss = 0
                log_info = {}
                
                main_cls = all_cls_tokens[:cum_sizes]
                
                target_proj_output = projector_dict['SimGCD'](main_cls)
                
                simgcd_cls_loss, simgcd_cluster_loss, simgcd_contrastive_loss, simgcd_sup_con_loss = simgcd_loss(
                    studdent_out_put=target_proj_output,
                    mask_lab=mask_lab,
                    class_labels=class_labels,
                    criterion=cluster_criterion,
                    epoch=epoch
                )
                
                total_loss += (1 - args.sup_weight) * simgcd_cluster_loss + args.sup_weight * simgcd_cls_loss
                total_loss += (1 - args.sup_weight) * simgcd_contrastive_loss + args.sup_weight * simgcd_sup_con_loss
                
                # 记录日志
                log_info.update({
                    'simgcd_cls_loss': simgcd_cls_loss.item(),
                    'simgcd_cluster_loss': simgcd_cluster_loss.item(),
                    'simgcd_sup_con_loss': simgcd_sup_con_loss.item(),
                    'simgcd_contrastive_loss': simgcd_contrastive_loss.item()
                })
                
                
                if args.use_contrastive_cluster:
                    for i, name in enumerate(dccl_heads):
                        dccl_cls = all_cls_tokens[cum_sizes:]
                        
                        dccl_proj_output = projector_dict[name](dccl_cls)
                        dccl_features = dccl_proj_output[0]  
                        
                        dccl_labels = dccl_labels_head[name]
                        dccl_contrastive_loss = memory_dict[name](dccl_features, dccl_labels)
                        
                        dccl_weight = DCCL_contrastive_cluster_weight_schedule[epoch]
                        total_loss += dccl_weight * dccl_contrastive_loss
                        
                        log_info[f'{name}_loss'] = dccl_contrastive_loss.item()
                        log_info[f'{name}_weight'] = dccl_weight
                        
                        if name in ['DCCL2', 'DCCL3']:
                            t_align_start = time.time()
                            
                            base_centers = memory_dict['DCCL'].features
                            dccl_centers = memory_dict[name].features
                            
                            base_features = projector_dict['DCCL'](dccl_cls)[0]  
                            base_features_norm = F.normalize(base_features, p=2, dim=1)
                            base_output = base_features_norm @ base_centers.t()
                            
                            align_loss = align_to_base_soft_js(
                                dccl_features, 
                                dccl_centers,
                                base_centers, 
                                base_output
                            )
                            
                            align_weight = DCCL_align_weight_schedule[epoch]
                            total_loss += align_weight * align_loss
                            
                            log_info[f'{name}_align_loss'] = align_loss.item()
                            log_info[f'{name}_align_weight'] = align_weight
                        
            
            loss_record.update(total_loss.item(), class_labels.size(0))
            
            optimizer.zero_grad()
            if fp16_scaler is None:
                total_loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(total_loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            
        
            if batch_idx % args.print_freq == 0:
                log_str = f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] '
                log_str += f'Loss: {total_loss.item():.5f} '
                wandb_log = {
                    'train/total_loss': total_loss.item(),
                    'train/batch_idx': batch_idx,
                    'epoch': epoch,
                    'global_step': epoch * len(train_loader) + batch_idx
                }

                for key, value in log_info.items():
                    if 'loss' in key:
                        log_str += f'{key}: {value:.4f} '
                        wandb_log[f'train/{key}'] = value
                
                args.logger.info(log_str)


        args.logger.info(f'Train Epoch: {epoch} Avg Loss: {loss_record.avg:.4f}')
        
        exp_lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            'train/learning_rate': current_lr,
            'epoch': epoch
        })
    
    
    return model, projector_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MGCE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # base params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--cluster_batch_size', default=2048, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--cluster_num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])
    
    # datasets 
    parser.add_argument('--dataset_name', type=str, default='cifar100',
                        help='options: hete_federated_cub cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', default=True, help="specific for cauterised")

    # training
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--hidden_dim', type=int, default=2048)

    # transform
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--n_views', default=2, type=int)

    # from simgce
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float,
                        help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                        help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature.')
    parser.add_argument('--sup_weight', type=float, default=0.35)

    # DCCL
    parser.add_argument('--use_contrastive_cluster', type=bool, default=True)
    parser.add_argument('--DCCL_contrastive_cluster_weight', type=float, default=0.1)
    parser.add_argument('--DCCL_align_weight', type=float, default=0.1)
    # parser.add_argument('--align_loss_weight', type=float, default=0.3)
    parser.add_argument('--contrastive_cluster_epochs', type=int, default=100, help=['a-1 -> a', 'a->a'])
    parser.add_argument('--dccl_update_freq', type=int, default=1, help="1 for finegrain, 5 for general")

    # infomap
    parser.add_argument('--k1', type=int, default=15,
                        help="hyperparameter for KNN")
    parser.add_argument('--k2', type=int, default=4,
                        help="hyperparameter for outline")
    parser.add_argument('--use-hard', default=False)
    parser.add_argument('--memory_momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--use_cluster_head', type=bool, default=False,
                        help="learning rate")
    parser.add_argument('--num_instances', type=int, default=16)
    parser.add_argument('--max_sim', type=bool, default=True)
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for adj")
    parser.add_argument('--feature_output_index', type=int, default=0,
                        help="0 for projector 2 for backbone")
    parser.add_argument('--single_minsim_1', type=float, default=0.6)
    parser.add_argument('--single_minsim_2', type=float, default=0.6)
    parser.add_argument('--single_minsim_3', type=float, default=0.6)
    parser.add_argument('--single_k1', type=int, default=10)
    parser.add_argument('--k1_ratio', type=float, default=0.6)

    # runs
    parser.add_argument('--runner_name', default='MGCE_debug', type=str)
    parser.add_argument('--exp_name', default='scars_test', type=str)
    parser.add_argument('--exp_root', type=str, default='')
    parser.add_argument('--cub_root', type=str, default='')
    parser.add_argument('--cifar_100_root', type=str, default='')
    parser.add_argument('--cars_root', type=str, default='')
    parser.add_argument('--pets_root', type=str, default='')
    parser.add_argument('--aircraft_root', type=str, default='')
    parser.add_argument('--inaturalist_root', type=str, default='')
    parser.add_argument('--imagenet_root', type=str, default='')
    parser.add_argument('--nabirds_root', type=str, default='')
    parser.add_argument('--herbarium_dataroot', type=str, default='')
    parser.add_argument('--osr_split_dir', type=str, default='')
    parser.add_argument('--dino_pretrain_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)

    # wandb
    parser.add_argument('--wandb_api_key', default="", type=str, help="WandB API Key")
    parser.add_argument('--wandb_project', default='', type=str, help="WandB project name")
    parser.add_argument('--wandb_entity', default='', type=str, help="WandB entity name")

    # Other params
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--use_gpu_knn', type=bool, default=True)
    parser.add_argument('--use_token_cache', type=bool, default=True)
    parser.add_argument('--token_cache_dir', type=str, default='')
    parser.add_argument('--cls_export_bs', type=int, default=2048)


    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ----------------------
    # SEED
    # ----------------------
    init_seed_torch(args.seed)

    # ----------------------
    # Get K1 and eps
    # ----------------------
    minsim_1 = args.single_minsim_1
    minsim_2 = args.single_minsim_2  
    minsim_3 = args.single_minsim_3
    k1_1 = args.single_k1
    k1_2 = round(args.single_k1 * args.k1_ratio)
    k1_3 = round(args.single_k1 / args.k1_ratio)

    args.k1_dict = {'DCCL': k1_1, 'DCCL2': k1_2, 'DCCL3': k1_3}
    args.eps_dict = {'DCCL': minsim_1, 'DCCL2': minsim_2, 'DCCL3': minsim_3}

    # ----------------------
    # Init
    # ----------------------
    init_experiment(args, runner_name=[f'{args.runner_name}'])

    # ----------------------
    # Datasets info
    # ----------------------
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.logger.info(f"Number of labeled classes: {args.num_labeled_classes}")
    args.logger.info(f"Number of unlabeled classes: {args.num_unlabeled_classes}")
    args.logger.info(f"Train classes: {args.train_classes}")
    args.logger.info(f"Unlabeled classes: {args.unlabeled_classes}")

    # ----------------------
    # Model
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    model = vits.__dict__['vit_base']()
    state_dict = torch.load(args.dino_pretrain_path, map_location='cpu')
    model.load_state_dict(state_dict)
    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    for m in model.parameters():
        m.requires_grad = False
    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in model.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True    

    model.to(args.device)

    args.dccl_heads = ['DCCL', 'DCCL2', 'DCCL3'] if args.use_contrastive_cluster else []

    projector_SimGCD = BranchHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projector_DCCL = BranchHead_New(in_dim=args.feat_dim, nlayers=2, bottleneck_dim=768, hidden_dim=args.hidden_dim)
    projector_DCCL2 = BranchHead_New(in_dim=args.feat_dim, nlayers=2, bottleneck_dim=768, hidden_dim=args.hidden_dim)
    projector_DCCL3 = BranchHead_New(in_dim=args.feat_dim, nlayers=2, bottleneck_dim=768, hidden_dim=args.hidden_dim)
    projector_dict = torch.nn.ModuleDict({"SimGCD": projector_SimGCD, "DCCL": projector_DCCL, "DCCL2": projector_DCCL2, "DCCL3": projector_DCCL3})

    # ----------------------
    # DATASETS and DATALOADERS
    # ----------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    args.dccl_transform = train_transform
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    train_dataset, cluster_dataset = get_datasets(args.dataset_name, train_transform, test_transform, args)
    args.logger.info(f"Train dataset size: {len(train_dataset)}")
    args.logger.info(f"Cluster dataset size: {len(cluster_dataset)}")
    args.logger.info(f"Labelled dataset size: {len(train_dataset.labelled_dataset)}")
    args.logger.info(f"Unlabelled dataset size: {len(train_dataset.unlabelled_dataset)}")

    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                                sampler=sampler, drop_last=True, pin_memory=True)
    
    cluster_loader = DataLoader(
        deepcopy(cluster_dataset),
        num_workers=getattr(args, "cluster_num_workers", 8),
        batch_size=getattr(args, "cluster_batch_size", 2048),
        shuffle=False, pin_memory=True, persistent_workers=True,
        prefetch_factor=6, drop_last=False, pin_memory_device="cuda:0",
    )

    wandb.init(
        project=args.wandb_project, 
        entity=args.wandb_entity, 
        name=args.exp_name,  
        config=args, 
    )

    train(model, projector_dict, train_loader, cluster_loader, args)

    wandb.finish()