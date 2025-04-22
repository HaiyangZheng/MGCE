#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python 9.HITSZ_explore_full_original.py \
 --dataset_name='scars' \
 --single_minsim_1=0.6 \
 --single_minsim_2=0.6 \
 --single_minsim_3=0.6 \
 --single_k1=15 \
 --k1_ratio=0.6 \
 --DCCL_contrastive_cluster_weight=0.1 \
 --align_loss_weight=0.1 \
 --seed=0 \
 --lr=0.05 \
 --exp_name='scars_hitsz_lr(0.05)_test1' \
 > log_explore/scars_hitsz_lr_test1.log 2>&1