import numpy as np
import random
import torch

# seed=0
# random.seed(seed)
# # os.environ["PYTHONHASHSEED"] = str(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# # torch.use_deterministic_algorithms(True)

import argparse
import math
import os
import sys
from collections import defaultdict

import torch.nn as nn
import wandb
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

# from data.augmentations import get_transform
# from data.get_datasets import get_datasets, get_class_splits
# 添加当前脚本目录的上一级目录的上一级目录的绝对路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
from data.augmentations import get_transform
from data.get_datasets_inaturalist import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config_iscap import exp_root, cub_root, cars_root, pets_root, osr_split_dir, dino_pretrain_path, cub_model_best, cifar_100_root, aircraft_root, herbarium_dataroot, imagenet_root
from model import BranchHead, BranchHead_New, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, \
    get_params_groups
from project_utils.contrastive_utils import extract_features, extract_features_with_headlist
from project_utils.infomap_cluster_utils import cluster_by_semi_infomap, get_dist_nbr, generate_cluster_features

from project_utils.cluster_memory_utils import ClusterMemory as DCCL_ClusterMemory
from project_utils.gcn_cluster_memory_utils import ClusterMemory

from project_utils.data_utils import IterLoader, FakeLabelDataset
from project_utils.sampler import RandomMultipleGallerySamplerNoCam
from copy import deepcopy
# from data.gcn_e_dataset import GCNEDataset
from models.gcn_e import GCN_E
# from data.utils.gcn_clustering import train_gcn, test_gcn_e
from sklearn.cluster import KMeans
import torch.nn.functional as F
from vpt_utils.model_create import create_backbone

# import ClusterEnsembles as CE

from scipy.optimize import linear_sum_assignment

def find_best_cluster_match(preds_dccl, preds_dccl2):
    # 确定每个聚类结果中簇的数量
    n_clusters_dccl = len(np.unique(preds_dccl))
    n_clusters_dccl2 = len(np.unique(preds_dccl2))
    
    # 创建成本矩阵
    cost_matrix = np.zeros((n_clusters_dccl, n_clusters_dccl2))
    for i in range(n_clusters_dccl):
        for j in range(n_clusters_dccl2):
            # 计算成本：这里使用的是不匹配的样本数量
            cost_matrix[i, j] = np.sum((preds_dccl == i) != (preds_dccl2 == j))
    
    # 执行匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 构建映射字典
    cluster_dict = {}
    for i, j in zip(col_ind, row_ind):
        cluster_dict[i] = j

    # 处理没有匹配的簇
    if n_clusters_dccl2 > n_clusters_dccl:
        unmatched_clusters = set(range(n_clusters_dccl2)) - set(col_ind)
        for cluster in unmatched_clusters:
            cluster_dict[cluster] = None

    return cluster_dict

def cluster_fusion(preds_dccl, preds_dccl2, preds_dccl3):
    # Find best matches
    cluster_dict1 = find_best_cluster_match(preds_dccl, preds_dccl2)
    cluster_dict2 = find_best_cluster_match(preds_dccl, preds_dccl3)
    
    # Initialize the result array
    final_preds = np.copy(preds_dccl)
    
    # Iterate over all predictions in preds_dccl
    for i in range(len(preds_dccl)):
        if preds_dccl[i] == -1:
            continue
        pred_dccl = preds_dccl[i]
        pred_dccl2 = cluster_dict1.get(preds_dccl2[i], None)
        pred_dccl3 = cluster_dict2.get(preds_dccl3[i], None)
        
        # Collect the predictions
        preds = [pred for pred in [pred_dccl, pred_dccl2, pred_dccl3] if pred is not None]
        
        # Decide the final cluster for this data point
        if pred_dccl2 is not None and preds.count(pred_dccl2) >= 2:
            final_preds[i] = pred_dccl2
            # print(preds)
        elif pred_dccl3 is not None and preds.count(pred_dccl3) >= 2:
            final_preds[i] = pred_dccl3
            # print(preds)
        
    return final_preds

def DCCL_Align_loss_with_head(name, DCCL_labels_base, features, memory_dict, temp):
    # 计算子集到中心的相似度矩阵
    sub_to_center_sim = memory_dict[name].features @ memory_dict['DCCL'].features.t()
    # 归一化 features
    norm_features = F.normalize(features, p=2, dim=1)  # L2归一化
    # 计算归一化特征和内存中特征的相似性
    similarity = norm_features.mm(memory_dict[name].features.t())
    # 将相似度结果除以温度参数 temp
    outputs = similarity / temp
    # 输出结果乘以子到中心的相似度矩阵
    outputs = outputs @ sub_to_center_sim
    # 计算交叉熵损失
    loss = F.cross_entropy(outputs, DCCL_labels_base)

    return loss

def align_loss_to_simgcd(DCCL_f_out, dccl_centers, simgcd_centers, simgcd_output):
    # dccl的中心是归一化过的，simgcd的中心是使用归一化的特征更新的，所以可以直接计算相似度
    # dccl的中心是没有梯度的，注册缓冲区的张量不参与梯度运算，但是simgcd中心有梯度，是否应该detach？
    dccl_to_simgcd_sim = dccl_centers @ simgcd_centers.t()
    # 归一化特征
    norm_features = F.normalize(DCCL_f_out, p=2, dim=1)
    # 计算归一化特征和dccl中心的相似性
    # similarity = norm_features.mm(dccl_centers.t())
    similarity = norm_features @ dccl_centers.t()

    outputs = similarity @ dccl_to_simgcd_sim

    pseudo_label = simgcd_output.argmax(dim=1)

    loss = F.cross_entropy(outputs, pseudo_label)
    return loss

def align_loss_to_simgcd_soft(DCCL_f_out, dccl_centers, simgcd_centers, simgcd_output):
    # dccl的中心是归一化过的，simgcd的中心是使用归一化的特征更新的，所以可以直接计算相似度
    # dccl的中心是没有梯度的，注册缓冲区的张量不参与梯度运算，但是simgcd中心有梯度，是否应该detach？
    dccl_to_simgcd_sim = dccl_centers @ simgcd_centers.t()
    # 归一化特征
    norm_features = F.normalize(DCCL_f_out, p=2, dim=1)
    # 计算归一化特征和dccl中心的相似性
    # similarity = norm_features.mm(dccl_centers.t())
    similarity = norm_features @ dccl_centers.t()

    outputs = similarity @ dccl_to_simgcd_sim

    outputs = F.softmax(outputs, dim=1)  # 将输出转换为概率分布

    # pseudo_label 是软标签，也是概率分布
    pseudo_label = F.softmax(simgcd_output, dim=1)

    # 计算 KL 散度
    # 注意: F.kl_div 需要对数概率作为输入，因此在 outputs 上使用 log_softmax
    log_prob_outputs = F.log_softmax(outputs, dim=1)
    loss = F.kl_div(log_prob_outputs, pseudo_label, reduction='batchmean')

    return loss

def align_loss_to_simgcd_soft_js(DCCL_f_out, dccl_centers, simgcd_centers, simgcd_output):
    # 计算两个中心之间的相似度
    dccl_to_simgcd_sim = dccl_centers @ simgcd_centers.t()
    # 归一化特征
    norm_features = F.normalize(DCCL_f_out, p=2, dim=1)
    # 计算归一化特征和dccl中心的相似性
    similarity = norm_features @ dccl_centers.t()

    outputs = similarity @ dccl_to_simgcd_sim
    outputs = F.softmax(outputs, dim=1)  # 将输出转换为概率分布
    pseudo_label = F.softmax(simgcd_output, dim=1)  # pseudo_label 是软标签，也是概率分布

    # 计算两个分布的对数概率
    log_prob_outputs = F.log_softmax(outputs, dim=1)
    log_prob_labels = F.log_softmax(pseudo_label, dim=1)

    # 计算两个方向的KL散度
    kl_div1 = F.kl_div(log_prob_outputs, pseudo_label, reduction='batchmean')
    kl_div2 = F.kl_div(log_prob_labels, outputs, reduction='batchmean')

    # 对称KL散度 (Jensen-Shannon divergence)
    js_divergence = 0.5 * (kl_div1 + kl_div2)

    return js_divergence

def test_with_pseudo_label_full(name, pred, targets, save_name='Train ACC Unlabelled', epoch=0):
    # 检查 pred 和 targets 长度是否相同
    if len(pred) != len(targets):
        raise ValueError("Length of predictions and targets must be the same.")
    
    # 打印 pred 的长度
    args.logger.info(f"Length of pred: {len(pred)}")
    
    # 根据 targets 计算 mask
    # mask = targets <= 100
    # mask = [target-1 in args.train_classes for target in targets]
    mask = [target in args.train_classes for target in targets]
    mask = np.array(mask).astype(bool)

    
    # 处理方式1：将 pred 中的 -1 替换为最大元素 + 1
    max_pred = np.max(pred)
    pred_1 = np.copy(pred)
    pred_1[pred_1 == -1] = max_pred + 1
    
    all_acc_1, old_acc_1, new_acc_1 = log_accs_from_preds(
        y_true=targets, y_pred=pred_1, mask=mask, T=epoch,
        eval_funcs=args.eval_funcs, save_name=save_name, args=args)
    
    args.logger.info(f"{name} Pseudo method 1 - All Acc: {all_acc_1}, Old Acc: {old_acc_1}, New Acc: {new_acc_1}")
    
    # 处理方式2：将 pred 中的 -1 依次替换为最大元素 + 1, 最大元素 + 2, ...
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
    
    args.logger.info(f"{name} Pseudo method 2 - All Acc: {all_acc_2}, Old Acc: {old_acc_2}, New Acc: {new_acc_2}")
    # 保存 pred 和 targets 到本地
    current_path = os.getcwd()
    pred_path = os.path.join(current_path, "pred.npy")
    targets_path = os.path.join(current_path, "targets.npy")
    
    # np.save(pred_path, pred)
    # np.save(targets_path, targets)

def test_with_pseudo_label(pseudo_labeled_dataset, save_name='Train ACC Unlabelled', epoch=0):
    targets, preds, mask = [], [], []
    for item_ in pseudo_labeled_dataset:
        path, label, ground_truth, if_labeled = item_[0], item_[1], item_[2], item_[3]
        targets.append(ground_truth)
        preds.append(label)
        mask.append(if_labeled)
    # preds, targets, mask = pred.detach(), ground_truth.detach(), if_labeled.detach()
    targets = np.array(targets)
    preds = np.array(preds)
    mask = np.array(mask)
    args.logger.info(f"Length of preds: {len(preds)}, Number of unique elements in preds: {len(np.unique(preds))}")
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


def cluster_align_loss(target_x, contexture_prototype, target_prototype, criterion, epoch, if_contexture_detach=True):
    if if_contexture_detach:
        contexture_prototype = contexture_prototype.detach()
    target_x = F.normalize(target_x, dim=-1, p=2)
    score_center = target_x @ contexture_prototype.t()
    score_center = F.softmax(score_center, dim=1)
    socre_relation = contexture_prototype @ target_prototype.t()  # equal to contrastive_centers @ self.last_layer.weight.t()
    student_out = score_center @ socre_relation
    teacher_out = student_out.detach()
    # clustering, sup
    # sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
    # sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
    # cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

    # clustering, unsup
    cluster_loss = criterion(student_out, teacher_out, epoch)
    return cluster_loss

# target_x：目标特征向量
def align_loss(target_x, contexture_prototype, target_prototype, mask_lab, class_labels, criterion, epoch, if_contexture_detach=True):
    if if_contexture_detach:
        contexture_prototype = contexture_prototype.detach()
    # 归一化
    target_x = F.normalize(target_x, dim=-1, p=2)
    # 计算目标特征向量 target_x 与 contexture_prototype 之间的内积
    score_center = target_x @ contexture_prototype.t()
    score_center = F.softmax(score_center, dim=1)
    
    # 计算 contexture_prototype 与 target_prototype 之间的内积
    socre_relation = contexture_prototype @ target_prototype.t()  # equal to contrastive_centers @ self.last_layer.weight.t()
    
    # student_out是借助contexture_prototype得到的logits
    student_out = score_center @ socre_relation
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
    loss = (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
    return loss


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


def DCCL(clustering_dataset, backbone, projector_dict, epoch, args):
    # 使用whole_train_test_dataset获得data loader
    cluster_loader = DataLoader(deepcopy(clustering_dataset), num_workers=args.num_workers,
                                batch_size=args.batch_size, shuffle=False)
    #?# 定义两个字典什么作用？
    contrastive_cluster_train_loader_predefine_dict, memory_dict = {}, {}
    with torch.no_grad():
        print('==> Create pseudo labels for unlabeled data')
        # cluster_loader = unlabelled_train_loader
        # cluster_loader = deepcopy(whole_train_test_loader)
        # 针对vision prompt，把每个prompt需要的全部特征、标签都提取整理好
        results_dict = extract_features_with_headlist(backbone, projector_dict, cluster_loader, args)

        preds_dict = {}
        targets_cs = []
        for name in projector_dict.keys():
            # simgcd的头不参与这一过程
            if "target" in name:
                continue

            features = results_dict[name + "_features"]
            label_mark = results_dict[name + "_labels"]
            # if_labeled = results_dict[name + "_indexs"]
            if_labeled = results_dict[name + "_if_labeded"]
            features_array = F.normalize(features, dim=1).cpu().numpy()
            # (num_samples, k)形状
            feat_dists, feat_nbrs = get_dist_nbr(features=features_array, k=args.k1_dict[name], knn_method='faiss-gpu',
                                                 device=0)

            # feature here output from layernorm without l2 norm
            # l2 norm for only calculate distance
            # min_sim用来控制建立边的最低相似性
            # cluster_num用来控制有效簇的最少数量
            # label_mark就是标签信息
            # if_labeled用来判断是否是有标签数据
            pseudo_labels = cluster_by_semi_infomap(feat_nbrs, feat_dists, min_sim=args.eps_dict[name], cluster_num=args.k2,
                                                    label_mark=label_mark, if_labeled=if_labeled, args=args)
            pseudo_labels = pseudo_labels.astype(np.intp)

            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

            del features_array
            # 得到簇中心
            cluster_features = generate_cluster_features(pseudo_labels, features)
            del features

            # Create hybrid memory
            memory = DCCL_ClusterMemory(num_features=args.feat_dim, num_samples=num_cluster, temp=args.temp,
                                        momentum=args.memory_momentum, use_hard=args.use_hard).to(args.device)

            # 簇中心归一化后赋值给memory.features
            memory.features = F.normalize(cluster_features, dim=1).to(args.device)
            # memory.feature  first without_l2 -> mean  -> l2 save
            if args.fp16:
                memory.features = memory.features.half()
                # trainer.memory = memory
            memory_dict[name] = memory

            # 构造pseudo_labeled_dataset
            pseudo_labeled_dataset = []

            item_targets_list = []

            for i, (_item, label, _if_labeled) in enumerate(zip(whole_train_test_dataset.data, pseudo_labels, if_labeled)):
                # if args.dataset_name == 'hete_federated_scars':
                #     item_targets_list.append(whole_train_test_dataset.targets[i]+1)
                # elif args.dataset_name == 'hete_federated_pets':
                #     item_targets_list.append(_item[2]+1)
                # else:
                #     item_targets_list.append(_item[2])
                if args.dataset_name != 'imagenet_100':
                    item_targets_list.append(whole_train_test_dataset.targets[i])
                else:
                    item_targets_list.append(_item[1])

                if label != -1:
                    if isinstance(_item, str):
                        pseudo_labeled_dataset.append((_item, label.item(), whole_train_test_dataset.targets[i], _if_labeled.item()))
                    elif args.dataset_name == 'imagenet_100':
                        pseudo_labeled_dataset.append((_item[0], label.item(), _item[1], _if_labeled.item()))                    
                    else:
                        pseudo_labeled_dataset.append((_item[1], label.item(), whole_train_test_dataset.targets[i], _if_labeled.item()))
                    # # if isinstance(_item, str):
                    # #     pseudo_labeled_dataset.append((_item, label.item(), _item, _if_labeled.item()))
                    # if args.dataset_name == 'imagenet_100':
                    #     pseudo_labeled_dataset.append((_item[0], label.item(), _item[1], _if_labeled.item()))
                    # elif args.dataset_name == 'hete_federated_scars':
                    #     pseudo_labeled_dataset.append((_item, label.item(), whole_train_test_dataset.targets[i]+1, _if_labeled.item()))
                    # elif args.dataset_name == 'hete_federated_pets':
                    #     pseudo_labeled_dataset.append((_item[1], label.item(), _item[2]+1, _if_labeled.item()))
                    # else:
                    #     pseudo_labeled_dataset.append((_item[1], label.item(), _item[2], _if_labeled.item()))

            item_targets = np.array(item_targets_list)

            len_of_labeled = len(whole_train_test_dataset.labelled_dataset)
            args.logger.info(f'length of whole_train_test_dataset.labeled_set {len_of_labeled}')
            test_with_pseudo_label_full(name, pred=pseudo_labels[len_of_labeled:], targets=item_targets[len_of_labeled:])

            # # 假设 pseudo_labels 和 item_targets 都是已经定义好的 numpy 数组
            # pseudo_labels = np.array(pseudo_labels[len_of_labeled:])
            # item_targets = np.array(item_targets[len_of_labeled:])
            # # 保存 pseudo_labels 数组
            # np.save(f'{name}_pseudo_labels.npy', pseudo_labels)
            # # 保存 item_targets 数组
            # np.save(f'{name}_item_targets.npy', item_targets)

            preds_dict[f'{name}'] = pseudo_labels[len_of_labeled:]
            targets_cs = item_targets[len_of_labeled:]

            args.logger.info(f'==> {name} Infomap for epoch {epoch}: {num_cluster} clusters')
            wandb.log({f"{name} infomap predicted number of cluster": num_cluster})

            # all_acc, old_acc, new_acc = test_with_pseudo_label(pseudo_labeled_dataset, save_name=f'{name} pseudo acc', epoch=epoch)
            # args.logger.info(f'{name} pseudo Accuracies: All {all_acc} | Old {old_acc} | New {new_acc}')

            # wandb.log(
            #     {f"{name} pseudo ALL": all_acc, f"{name} pseudo OLD": old_acc, f"{name} pseudo NEW": new_acc})

            # pseudo_labeled_dataset 是伪标签数据集 args.num_instances 指定了每个类别中要采样的实例数量
            PK_sampler = RandomMultipleGallerySamplerNoCam(pseudo_labeled_dataset, args.num_instances)

            # image_dir = os.path.join(unlabelled_train_examples_test.root, unlabelled_train_examples_test.base_folder)
            contrastive_cluster_train_loader = IterLoader(
                DataLoader(FakeLabelDataset(pseudo_labeled_dataset, root=None, transform=train_transform),
                           batch_size=args.batch_size, num_workers=args.num_workers, sampler=PK_sampler,
                           shuffle=False, pin_memory=True, drop_last=True))
            contrastive_cluster_train_loader.new_epoch()
            contrastive_cluster_train_loader_predefine_dict[name] = contrastive_cluster_train_loader

        # labels = np.array([preds_dict['DCCL'], preds_dict['DCCL2'], preds_dict['DCCL3']])
        # label_ce = CE.cluster_ensembles(labels, solver='mcla')
        # test_with_pseudo_label_full('Ensemble clustering', pred=label_ce, targets=targets_cs)
        cluster_fusion_labels = cluster_fusion(preds_dict['DCCL'], preds_dict['DCCL2'], preds_dict['DCCL3'])
        test_with_pseudo_label_full('Ensemble clustering', pred=cluster_fusion_labels, targets=targets_cs)

        # labels = np.array([preds_dict['DCCL'], preds_dict['DCCL2'], preds_dict['DCCL3']])
        # label_ce = CE.cluster_ensembles(labels, solver='mcla', nclass=args.mlp_out_dim)
        # test_with_pseudo_label_full('True class Ensemble clustering', pred=label_ce, targets=targets_cs)

        # ###added by haiyang for backbone feature test
        # features = results_dict['backbone' + "_features"]
        # label_mark = results_dict['backbone' + "_labels"]
        # if_labeled = results_dict['backbone' + "_if_labeded"]
        # features_array = F.normalize(features, dim=1).cpu().numpy()
        # # (num_samples, k)形状
        # feat_dists, feat_nbrs = get_dist_nbr(features=features_array, k=args.k1_dict['DCCL'], knn_method='faiss-gpu',
        #                                         device=0)
        # pseudo_labels = cluster_by_semi_infomap(feat_nbrs, feat_dists, min_sim=args.eps_dict['DCCL'], cluster_num=args.k2,
        #                                         label_mark=label_mark, if_labeled=if_labeled, args=args)
        # pseudo_labels = pseudo_labels.astype(np.intp)
        # num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        # del features_array
        # del features
        # # 构造pseudo_labeled_dataset
        # pseudo_labeled_dataset = []
        # item_targets_list = []
        # for i, (_item, label, _if_labeled) in enumerate(zip(whole_train_test_dataset.data, pseudo_labels, if_labeled)):
        #     # if args.dataset_name == 'hete_federated_scars':
        #     #     item_targets_list.append(whole_train_test_dataset.targets[i]+1)
        #     # else:
        #     #     item_targets_list.append(_item[2])
        #     item_targets_list.append(whole_train_test_dataset.targets[i])

        #     if label != -1:
        #         if isinstance(_item, str):
        #             pseudo_labeled_dataset.append((_item, label.item(), whole_train_test_dataset.targets[i], _if_labeled.item()))
        #         else:
        #             pseudo_labeled_dataset.append((_item[1], label.item(), whole_train_test_dataset.targets[i], _if_labeled.item()))

        #         # if isinstance(_item, str):
        #         #     pseudo_labeled_dataset.append((_item, label.item(), _item, _if_labeled.item()))
        #         # elif args.dataset_name == 'imagenet_100':
        #         #     pseudo_labeled_dataset.append((_item[0], label.item(), _item[1], _if_labeled.item()))
        #         # elif args.dataset_name == 'hete_federated_scars':
        #         #     pseudo_labeled_dataset.append((_item, label.item(), whole_train_test_dataset.targets[i]+1, _if_labeled.item()))
        #         # else:
        #         #     pseudo_labeled_dataset.append((_item[1], label.item(), _item[2], _if_labeled.item()))

        # item_targets = np.array(item_targets_list)

        # len_of_labeled = len(whole_train_test_dataset.labelled_dataset)
        # args.logger.info(f'length of whole_train_test_dataset.labeled_set {len_of_labeled}')
        # test_with_pseudo_label_full('backbone', pred=pseudo_labels[len_of_labeled:], targets=item_targets[len_of_labeled:])

        # args.logger.info(f'==> backbone Infomap for epoch {epoch}: {num_cluster} clusters')
        # wandb.log({f"backbone infomap predicted number of cluster": num_cluster})

    return contrastive_cluster_train_loader_predefine_dict, memory_dict


def train(backbone, projector_dict, train_loader, unlabelled_train_loader, args):
    backbone.to(args.device)
    projector_dict.to(args.device)

    params_groups = get_params_groups(backbone)
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

    # DCCL损失权重，前100epoch中从0.03->0.3，后100保持0.3
    DCCL_contrastive_cluster_weight_schedule = np.concatenate((
        np.linspace(args.DCCL_contrastive_cluster_weight * 0.1,
                    args.DCCL_contrastive_cluster_weight, args.contrastive_cluster_epochs),
        np.ones(args.epochs - args.contrastive_cluster_epochs) * args.DCCL_contrastive_cluster_weight
    ))

    DCCL_align_weight_schedule = np.concatenate((
        np.linspace(args.align_loss_weight * 0.1,
                    args.align_loss_weight, 100),
        np.ones(100) * args.align_loss_weight
    ))

    # 对齐损失权重
    # 初始：0-0 权重为0
    # 增加：0-100 权重0增加到1
    # 稳定：100-200 权重保持1
    align_loss_weight_schedule = np.concatenate((np.ones(args.align_loss_epoch_start) * args.align_loss_weight_start,
                                                 np.linspace(args.align_loss_weight_start,
                                                             args.align_loss_weight_end,
                                                             args.align_loss_epoch_smooth),
                                                 np.ones(
                                                     args.epochs - args.align_loss_epoch_smooth - args.align_loss_epoch_start) * args.align_loss_weight_end
                                                 ))

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        if args.use_contrastive_cluster:
            if epoch%5==0:
                DCCL_contrastive_cluster_train_loader_predefine_dict, memory_dict = DCCL(
                    clustering_dataset=whole_train_test_dataset,
                    backbone=backbone, projector_dict=projector_dict, epoch=epoch, args=args)
            # test_with_head(backbone=backbone, projector_dict=projector_dict, test_loader=unlabelled_train_loader,
            #                epoch=epoch,
            #                save_name='Train ACC Unlabelled',
            #                args=args,
            #                memory_dict=memory_dict)
        backbone.train()
        projector_dict.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab, _ = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):

                # outputs = backbone(images, return_all_patches=False)
                outputs = backbone(images, return_all_patches=True)
                loss = 0
                pstr = ''
                for i_, (name, projector) in enumerate(projector_dict.items()):

                    if "target" in name:
                        # proj_outputs = projector(outputs[:, i_, :])
                        proj_outputs = projector(outputs[:, 0, :])
                        # proj_outputs = projector(outputs)
                        # simgcd loss
                        target_cls_loss, target_cluster_loss, target_contrastive_loss, target_sup_con_loss = simgcd_loss(
                            studdent_out_put=proj_outputs, mask_lab=mask_lab,
                            class_labels=class_labels, criterion=cluster_criterion, epoch=epoch)
    
                        pstr += f'{name}_cls_loss: {target_cls_loss.item():.4f} '
                        pstr += f'{name}_cluster_loss: {target_cluster_loss.item():.4f} '
                        pstr += f'{name}_sup_con_loss: {target_sup_con_loss.item():.4f} '
                        pstr += f'{name}_contrastive_loss: {target_contrastive_loss.item():.4f} '
                        if batch_idx % args.print_freq == 0:
                            wandb.log(
                                {f"classifier_sup": target_cls_loss.item(), "classifier_unsup": target_cluster_loss.item(),
                                f"contrastive_sup": target_sup_con_loss.item(),
                                'contrastive_unsup': target_contrastive_loss.item(),
                                f"current_align_loss_weight": align_loss_weight_schedule[epoch]})

                        loss += (1 - args.sup_weight) * target_cluster_loss + args.sup_weight * target_cls_loss
                        loss += (1 - args.sup_weight) * target_contrastive_loss + args.sup_weight * target_sup_con_loss
                    # else:
                    #     loss += (1 - args.sup_weight) * target_cluster_loss + args.sup_weight * target_cls_loss
                    #     loss += (1 - args.sup_weight) * target_contrastive_loss + args.sup_weight * target_sup_con_loss

                    # clustering contrastive
                    if args.use_contrastive_cluster and "target" not in name:

                        DCCL_images, DCCL_labels, DCCL_indexes = DCCL_contrastive_cluster_train_loader_predefine_dict[
                            name].next()
                        if isinstance(DCCL_images, (list, tuple)):  # concat multiview
                            DCCL_images = torch.cat(DCCL_images, dim=0).to(args.device)  # [B*2/num_experts,3,224,224]
                            DCCL_labels2 = DCCL_labels.detach().clone()
                            DCCL_labels = torch.cat((DCCL_labels, DCCL_labels2), dim=0).to(args.device)
                        else:
                            DCCL_images = DCCL_images.to(args.device)
                            DCCL_labels = DCCL_labels.to(args.device)

                        # forward
                        # DCCL_output = backbone(DCCL_images, return_all_patches=False)
                        DCCL_output = backbone(DCCL_images, return_all_patches=True)
                        # DCCL_output = backbone(DCCL_images, True)
                        # DCCL_projector_output = projector(DCCL_output)
                        # # 对应visual prompt
                        # DCCL_projector_output = projector(DCCL_output[:, i_, :])
                        # 没有visual prompt的cls token
                        DCCL_projector_output = projector(DCCL_output[:, 0, :])
                        # args.feature_output_index=2 返回cls token
                        # DCCL_f_out = DCCL_projector_output[args.feature_output_index]
                        DCCL_f_out = DCCL_projector_output[0]

                        # 这里特征应该需要归一化？

                        DCCL_contrastive_cluster_loss = memory_dict[name](DCCL_f_out, DCCL_labels)

                        loss += DCCL_contrastive_cluster_weight_schedule[epoch] * DCCL_contrastive_cluster_loss.to(args.device)
                        # loss += DCCL_align_weight_schedule[epoch] * DCCL_contrastive_cluster_loss.to(args.device)
                        pstr += f'{name}_loss: {DCCL_contrastive_cluster_loss.item():.4f} '
                        if batch_idx % args.print_freq == 0:
                            wandb.log(
                                {f"{name} DCCL loss": DCCL_contrastive_cluster_loss.item()})
                            wandb.log(
                                {f"{name} current_DCCL_contrastive_cluster_weight":
                                     DCCL_contrastive_cluster_weight_schedule[
                                         epoch]})
                            
                        ### added by haiyang for align only to dccl
                        if name == 'DCCL2' or name == 'DCCL3':
                            dccl_centers = memory_dict[name].features
                            base_centers = memory_dict['DCCL'].features
                            # 手动重构权重
                            # simgcd_last_layer = projector_dict['target'].last_layer
                            # simgcd_centers = simgcd_last_layer.weight_g * simgcd_last_layer.weight_v.div(simgcd_last_layer.weight_v.norm(dim=0, keepdim=True))
                            base_feature = projector_dict['DCCL'](DCCL_output[:, 0, :])[0]
                            base_norm_features = F.normalize(base_feature, p=2, dim=1)
                            base_output = base_norm_features @ base_centers.t()

                            DCCL_align_loss = align_loss_to_simgcd_soft_js(DCCL_f_out,dccl_centers,base_centers, base_output)
                            loss += DCCL_align_weight_schedule[epoch] * DCCL_align_loss
                            pstr += f'{name}_align_loss: {DCCL_align_loss.item():.4f} '
                        ### added by haiyang for align only to dccl

                        # ###added by haiyang for DCCL_align_loss
                        # dccl_centers = memory_dict[name].features
                        # # 获取修饰后的线性层
                        # simgcd_centers = projector_dict['target'].last_layer.weight.data
                        # # 手动重构权重
                        # # simgcd_last_layer = projector_dict['target'].last_layer
                        # # simgcd_centers = simgcd_last_layer.weight_g * simgcd_last_layer.weight_v.div(simgcd_last_layer.weight_v.norm(dim=0, keepdim=True))
                        # simgcd_output = projector_dict['target'](outputs[:, 0, :])[1]
                        # DCCL_align_loss = align_loss_to_simgcd_soft(DCCL_f_out,dccl_centers,simgcd_centers, simgcd_output)
                        # loss += DCCL_align_weight_schedule[epoch] * DCCL_align_loss
                        # pstr += f'{name}_align_loss: {DCCL_align_loss.item():.4f} '
                        # # loss += 0.5 * DCCL_align_loss
                        # ###added by haiyang for DCCL_align_loss

                        ##### align
                        # DCCL_align_loss = align_loss(target_x=target_class_token,
                        # DCCL_align_loss = align_loss(target_x=outputs[:, i_, :],
                        #                              contexture_prototype=memory_dict[name].features,
                        #                              target_prototype=projector_dict['target'].last_layer.weight,
                        #                              mask_lab=mask_lab, class_labels=class_labels,
                        #                              criterion=cluster_criterion, epoch=epoch,
                        #                              if_contexture_detach=False)
                        # DCCL_align_loss = align_loss(target_x=outputs[:, i_, :],
                        #                              contexture_prototype=projector_dict[name].last_layer.weight,
                        #                              target_prototype=projector_dict['target'].last_layer.weight,
                        #                              mask_lab=mask_lab, class_labels=class_labels,
                        #                              criterion=cluster_criterion, epoch=epoch,
                        #                              if_contexture_detach=False)
                        #
                        # DCCL_cluster_align_loss = cluster_align_loss(target_x=DCCL_output[:, i_, :],
                        #                                              contexture_prototype=projector_dict[name].last_layer.weight,
                        #                                              target_prototype=projector_dict[
                        #                                                  'target'].last_layer.weight,
                        #                                              criterion=cluster_criterion, epoch=epoch,
                        #                                              if_contexture_detach=False)

                        # loss += align_loss_weight_schedule[epoch] * (DCCL_align_loss )

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                                 .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))
        if epoch % args.test_interval == 0:

            args.logger.info('Testing on unlabelled examples in the training data...')
            wandb.log({f"Train Epoch": epoch, "Avg_loss": loss_record.avg})
            # test_with_head(backbone=backbone, projector_dict=projector_dict, test_loader=unlabelled_train_loader,
            #                epoch=epoch,
            #                save_name='Train ACC Unlabelled',
            #                args=args,
            #                memory_dict=memory_dict)

            save_dict = {
                'model': backbone.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }

            # if epoch == 99:
            #     torch.save(save_dict, args.model_path + ".99ep")
            #     args.logger.info("model saved to {}.".format(args.model_path) + ".99ep")
            # else:
            #     torch.save(save_dict, args.model_path)
            #     args.logger.info("model saved to {}.".format(args.model_path))

        # Step schedule
        exp_lr_scheduler.step()


def test(model, test_loader, epoch, save_name, args):
    model.eval()

    preds, targets, fused_preds = [], [], []
    mask = np.array([])
    for batch_idx, _item in enumerate(tqdm(test_loader)):
        images = _item[0].cuda(non_blocking=True)
        label = _item[1]
        with torch.no_grad():
            output_ = model(images)
            logits = output_[1]
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask,
                             np.array([True if x.item() in args.train_classes else False for x in label]))

    preds = np.concatenate(preds)

    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc





# def test_with_head(backbone, projector_dict, test_loader, epoch, save_name, args, memory_dict=None):
#     def _test_with_memory(query, memory_prototype, target_prototype):
#         target_x = query.detach()
#         norm_target = F.normalize(target_prototype.detach(), dim=-1, p=2)
#         contexture_prototype = F.normalize(memory_prototype.detach().float(), dim=-1, p=2)
#         target_x = F.normalize(target_x, dim=-1, p=2)
#         score_center = target_x @ contexture_prototype.t()
#         score_center = F.softmax(score_center, dim=1)
#         socre_relation = contexture_prototype @ norm_target.t()  # equal to contrastive_centers @ self.last_layer.weight.t()
#         socre_relation = F.softmax(socre_relation, dim=1)
#         student_out = score_center @ socre_relation
#         predict = F.softmax(student_out, dim=1)
#         return predict.detach()

#     backbone.eval()
#     projector_dict.eval()

#     results_dict = defaultdict(list)
#     mask = np.array([])

#     for batch_idx, _item in enumerate(tqdm(test_loader)):
#         images = _item[0].cuda(non_blocking=True)
#         label = _item[1]
#         mask = np.append(mask,
#                          np.array([True if x.item() in args.train_classes else False for x in label]))
#         with torch.no_grad():
#             # output_ = backbone(images, True)
#             output_ = backbone(images, return_all_patches=True)
#             for i_, (name, projector) in enumerate(projector_dict.items()):
#                 # proj_output_ = projector(output_)
#                 proj_output_ = projector(output_[:, i_, :])
#                 # if memory_dict is not None and 'target' not in name:
#                 #     memory_logits = _test_with_memory(output_[:, i_, :], memory_dict[name].features, projector_dict['target'].last_layer.weight)
#                 #     results_dict[name + "_memory_preds"].append(memory_logits.argmax(1).cpu().numpy())
#                 #     results_dict[name + "_memory_targets"].append(label.cpu().numpy())
#                 logits = proj_output_[1]
#                 prob = F.softmax(logits, dim=1)
#                 results_dict[name + "_preds"].append(prob.argmax(1).cpu().numpy())
#                 results_dict[name + "_targets"].append(label.cpu().numpy())


#     for i_, (name, projector) in enumerate(projector_dict.items()):
#         results_dict[name + "_preds"] = np.concatenate(results_dict[name + "_preds"])
#         results_dict[name + "_targets"] = np.concatenate(results_dict[name + "_targets"])
#         results_dict[name + "_mask"] = mask
#         # if memory_dict is not None and 'target' not in name:
#         #     results_dict[name + "_memory_preds"] = np.concatenate(results_dict[name + "_memory_preds"])
#         #     results_dict[name + "_memory_targets"] = np.concatenate(results_dict[name + "_memory_targets"])
#         #     results_dict[name + "_memory_mask"] = mask
#         #     memory_all_acc, memory_old_acc, memory_new_acc = log_accs_from_preds(y_true=results_dict[name + "_memory_targets"],
#         #                                                     y_pred=results_dict[name + "_memory_preds"],
#         #                                                     mask=results_dict[name + "_memory_mask"],
#         #                                                     T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
#         #                                                     args=args)

#         #     args.logger.info(f'memory_{name} Train Accuracies: All {memory_all_acc} | Old {memory_old_acc} | New {memory_new_acc}')

#         #     wandb.log(
#         #         {f"memory_{name} ALL": memory_all_acc, f"memory_{name} OLD": memory_old_acc, f"memory_{name} NEW": memory_new_acc})

#         all_acc, old_acc, new_acc = log_accs_from_preds(y_true=results_dict[name + "_targets"], y_pred=results_dict[name + "_preds"], mask=results_dict[name + "_mask"],
#                                                         T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
#                                                         args=args)

#         args.logger.info(f'{name} Length: {len(results_dict[name + "_preds"])} Train Accuracies: All {all_acc} | Old {old_acc} | New {new_acc}')

#         wandb.log(
#             {f"{name} ALL": all_acc, f"{name} OLD": old_acc, f"{name} NEW": new_acc})
        
    # combine_pred = results_dict['target' + "_preds"]
    # for i_, (name, projector) in enumerate(projector_dict.items()):
    #     if 'target' not in name:
    #         combine_pred += results_dict[name + "_preds"]
    # combine_pred = combine_pred / len(projector_dict.keys())
    # combine_all_acc, combine_old_acc, combine_new_acc = log_accs_from_preds(y_true=results_dict['target' + "_targets"],
    #                                                 y_pred=combine_pred,
    #                                                 mask=results_dict['target' + "_mask"],
    #                                                 T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
    #                                                 args=args)

    # args.logger.info(f'combine Train Accuracies: All {combine_all_acc} | Old {combine_old_acc} | New {combine_new_acc}')

    # wandb.log(
    #     {f"combine ALL": combine_all_acc, f"combine OLD": combine_old_acc, f"combine NEW": combine_new_acc})

def init_seed_torch(seed=1):
    random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

def get_worker_init_fn(initial_seed):
    def worker_init_fn(worker_id):
        seed = initial_seed + worker_id
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    return worker_init_fn

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])
    # parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='cifar100',
                        help='options: hete_federated_cub cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', default=True, help="specific for cauterised")

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)

    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float,
                        help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                        help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='cub_simGCD', type=str)
    # fed
    # federated argument heto
    parser.add_argument('--n_clients', type=int, default=1, help="0 or 1 for centralized learning")
    parser.add_argument('--dirichlet', type=float, default=0.2,
                        help="degree of heto")
    parser.add_argument('--rounds', type=int, default=200,
                        help="number of rounds of training")
    parser.add_argument('--local_train_ep', type=int, default=1,
                        help="the number of local episodes: E")
    parser.add_argument('--rate_local_label_unlabel', type=float, default=0.5)
    # GMM
    parser.add_argument('--show_client_data_info', type=bool, default=False)
    parser.add_argument('--use_gmm', type=bool, default=False)
    parser.add_argument('--use_gmm_agg', type=bool, default=False)
    parser.add_argument('--use_gmm_dual', type=bool, default=False)
    parser.add_argument('--begin_agg_round', type=int, default=0)
    parser.add_argument('--gmm_alpha', type=float, default=0.1)
    parser.add_argument('--weight_likelihood', type=float, default=0.01)
    parser.add_argument('--gmm_lr', type=float, default=0.1)
    parser.add_argument('--gmm_lr_multiply', type=float, default=0)
    parser.add_argument('--sampling_times', type=int, default=0)
    parser.add_argument('--local_global_contrastive_cluster_weight', type=float, default=0)

    parser.add_argument('--local_GMM_project_head', type=bool, default=False)
    parser.add_argument('--global_GMM_project_head', type=bool, default=False)

    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--cub_root', type=str, default=cub_root)
    # parser.add_argument('--cifar_10_root', type=str, default=cifar_10_root)
    parser.add_argument('--cifar_100_root', type=str, default=cifar_100_root)

    parser.add_argument('--hyper_k', type=int, default=231)
    parser.add_argument('--cars_root', type=str, default=cars_root)
    parser.add_argument('--pets_root', type=str, default=pets_root)
    parser.add_argument('--aircraft_root', type=str, default=aircraft_root)
    # parser.add_argument('--imagenet_root', type=str, default=imagenet_root)
    parser.add_argument('--herbarium_dataroot', type=str, default=herbarium_dataroot)
    parser.add_argument('--imagenet_root', type=str, default=imagenet_root)
    parser.add_argument('--osr_split_dir', type=str, default=osr_split_dir)
    parser.add_argument('--test_interval', type=int, default=1)

    # for DCCL
    parser.add_argument('--use_contrastive_cluster', type=bool, default=True)
    parser.add_argument('--GCN_contrastive_cluster_weight', type=float, default=0.3)
    parser.add_argument('--DCCL_contrastive_cluster_weight', type=float, default=0.3)
    parser.add_argument('--align_loss_weight', type=float, default=0.3)
    parser.add_argument('--contrastive_cluster_epochs', type=int, default=100, help=['a-1 -> a', 'a->a'])

    parser.add_argument('--align_loss_epoch_start', type=int, default=0)
    parser.add_argument('--align_loss_epoch_smooth', type=int, default=100)
    parser.add_argument('--align_loss_weight_start', type=float, default=0)
    parser.add_argument('--align_loss_weight_end', type=float, default=1)



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
    parser.add_argument('--feature_output_index', type=int, default=2,
                        help="-1 for backbone 2 for multi head projector")
    parser.add_argument('--wandb_offline', type=bool, default=True,
                        help="-1 for backbone 2 for multi head projector")

    # GCN
    parser.add_argument('--gcn_k1', type=int, default=80,
                        help="hyperparameter for KNN")
    parser.add_argument('--max_conn', type=int, default=1,
                        help="loss weight for graph")
    parser.add_argument('--gcn_train_epoch', type=int, default=40,
                        help="train epoch for graph")
    parser.add_argument('--use_sym', type=bool, default=False)
    parser.add_argument('--neg_size', default=10, type=int)

    # prompt
    # use_vpt控制是否使用vision prompt，在create_model时起作用
    parser.add_argument('--use_vpt', type=bool, default=False)
    parser.add_argument('--dino_pretrain_path', type=str, default=dino_pretrain_path)
    parser.add_argument('--load_from_model', type=str, default=cub_model_best)
    # parser.add_argument('--use_vpt', type=str2bool, default=True)
    parser.add_argument('--vpt_dropout', type=float, default=0.0)
    # 额外增加的prompt张量
    parser.add_argument('--num_prompts', type=int, default=5)  ### number of total prompts
    parser.add_argument('--predict_token', type=str, default='cop')
    # 这个参数涉及visual prompt内部结构，暂时不研究
    parser.add_argument('--n_shallow_prompts', type=int, default=0)  ### number of SHALLOW prompts prepended
    parser.add_argument('--num_dpr', type=int, default=4)  ### number of supervised prompts
    parser.add_argument('--w_prompt_clu', type=float, default=0.35)  ### DPR loss weight
    parser.add_argument('--unfrezee_only_last_prompt', type=bool, default=True)  ### DPR loss weight
    parser.add_argument('--single_minsim_1', type=float, default=0.6)
    parser.add_argument('--single_minsim_2', type=float, default=0.6)
    parser.add_argument('--single_minsim_3', type=float, default=0.6)
    parser.add_argument('--single_k1', type=int, default=10)
    parser.add_argument('--k1_ratio', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--single_k1_2', type=int, default=10)
    # parser.add_argument('--single_k1_3', type=int, default=10)
    # parser.add_argument('--return_all_patches', type=bool, default=False)  ### DPR loss weight
    # parser.add_argument('--cub_model_best', default=cub_model_best, type=str)
    # parser.add_argument('--aircraft_model_best', default=aircraft_model_best, type=str)
    # parser.add_argument('--cifar100_model_best', default=cifar100_model_best, type=str)
    # parser.add_argument('--cifar10_model_best', default=cifar10_model_best, type=str)
    # parser.add_argument('--herb_model_best', default=herb_model_best, type=str)
    # parser.add_argument('--imagenet_model_best', default=imagenet_model_best, type=str)
    # parser.add_argument('--scars_model_best', default=scars_model_best, type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    args.device = torch.device('cuda:0')

    # ----------------------
    # SEED
    # ----------------------
    init_seed_torch(args.seed)

    #?# 这个参数什么意思：k1指的应该是k近邻
    minsim_1 = args.single_minsim_1
    minsim_2 = args.single_minsim_2
    minsim_3 = args.single_minsim_3
    k1_1 = args.single_k1
    k1_2 = math.ceil(args.single_k1 * args.k1_ratio)
    k1_3 = math.ceil(args.single_k1 / args.k1_ratio)

    args.k1_dict = {'DCCL': k1_1, 'DCCL2': k1_2, 'DCCL3': k1_3, 'DCCL4': 80, 'DCCL5': 80}
    args.eps_dict = {'DCCL': minsim_1, 'DCCL2': minsim_2, 'DCCL3': minsim_3, 'DCCL4': 0.6, 'DCCL5': 0.5}

    # 脚本文件bash中不会指定datasetname，所以，用的就是default
    # hete_federated_cub设置和cub一致，名字是否有额外的含义
    args = get_class_splits(args)
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    # else:

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    # 这里修改log目录名称和实验名称: exp_root/runner_name/log/exp_name
    # args.exp_name='DCCL_align_threelevel'
    init_experiment(args, runner_name=['DCCL_align_loss'])
    args.logger.info(f'k1_2: {k1_2}, k1_3: {k1_3}')
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # wandb设置
    wandb.login(key="64c8de3836a70be723fc77af3b96d402ecba4fbe", timeout=60)
    run = wandb.init(
        # Set the project where this run will be logged
        project="DCCL_align_base",
        # Track hyperparameters and run metadata
        config=vars(args),
        name=args.log_dir + os.path.basename(__file__)
    )

    # cudnn设置为基准模式（每次遇到新的卷积层配置时，自动寻找和选择最快的算法来执行卷积操作）
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    # use vpt默认为True
    model = create_backbone(args, args.device)
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    """features = model(x, True)
            features = [features[:, 0, :], features[:, 1:1+num_prompts, :]]
            
            z_features = features[0]
            z_features = F.normalize(z_features, dim=-1)
            
            features[0] = projection_head(features[0])
            features[0] = F.normalize(features[0], dim=-1)
            
            prompt_features = features[1][:, :num_cop, :].mean(dim=1)
            z_prompt_features = F.normalize(prompt_features, dim=-1)
            prompt_features = kwargs['aux_projection_head'](prompt_features)
            prompt_features = F.normalize(prompt_features, dim=-1)
            if return_z_features:
                features = [features[0], z_features, prompt_features, z_prompt_features]
            else:
                features = [features[0], prompt_features]
            return features
    """
    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------

    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    # train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
    #                                                                                      train_transform,
    #                                                                                      test_transform,
    #                                                                                      args)
    # train_dataset, test_dataset, unlabelled_train_examples_test, unlabelled_train_examples_train, datasets, labelled_train_examples, \
    # whole_train_test_dataset, federated_train_datasets_dict, testset_labelled, testset_unlabelled = get_datasets(
    #     args.dataset_name,
    #     train_transform,
    #     test_transform,
    #     args)
    train_dataset, test_dataset, unlabelled_train_examples_test, unlabelled_train_examples_train, datasets, labelled_train_examples, \
    whole_train_test_dataset = get_datasets(
        args.dataset_name,
        train_transform,
        test_transform,
        args)
    # train_dataset, test_dataset, unlabelled_train_examples_test, datasets, whole_train_test_dataset = get_datasets(
    #     args.dataset_name,
    #     train_transform,
    #     test_transform,
    #     args)
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    ##
    # print("unlabelled_train_examples_test.target_transform", unlabelled_train_examples_test.dataset.target_transform)
    # print("unlabelled_train_examples_test.targets", unlabelled_train_examples_test.dataset.targets)
    # print("unlabelled_train_examples_test.targets.max()",max(unlabelled_train_examples_test.dataset.targets))
    # print("unlabelled_train_examples_test.targets.min()",min(unlabelled_train_examples_test.dataset.targets))
    # print("args.train_classes.max()",max(args.train_classes))
    # print("args.train_classes.min()",min(args.train_classes))
    ##
    # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    # 结构和simgcd中的DINO head相同，返回结果为return x_proj, logits, x
    projector_target = BranchHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    # projector_target = BranchHead(in_dim=args.feat_dim, out_dim=175, nlayers=args.num_mlp_layers)
    # projector_DCCL = BranchHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    # projector_DCCL2 = BranchHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    # projector_DCCL3 = BranchHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projector_DCCL = BranchHead_New(in_dim=args.feat_dim, nlayers=1, bottleneck_dim=768)
    projector_DCCL2 = BranchHead_New(in_dim=args.feat_dim, nlayers=1, bottleneck_dim=768)
    projector_DCCL3 = BranchHead_New(in_dim=args.feat_dim, nlayers=1, bottleneck_dim=768)
    # projector_DCCL4 = BranchHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    # projector_DCCL5 = BranchHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projector_dict = torch.nn.ModuleDict({"target": projector_target, "DCCL": projector_DCCL, "DCCL2": projector_DCCL2, "DCCL3": projector_DCCL3})
    # projector_dict = torch.nn.ModuleDict({"target": projector_target, "DCCL": projector_DCCL})
    # projector_dict = torch.nn.ModuleDict({"target": projector_target,
    #                                       "DCCL": projector_DCCL,
    #                                       "DCCL2": projector_DCCL2,
    #                                       "DCCL3": projector_DCCL3})

    # projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    # model = nn.Sequential(backbone, projector).to(device)
    # Clustering_Contrastive_Head(in_dim=args.feat_dim, out_dim=args.feat_dim, nlayers=args.num_mlp_layers)
    # print(projector)
    # ----------------------
    # TRAIN
    # ----------------------
    # train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    train(model, projector_dict, train_loader, test_loader_unlabelled, args)
