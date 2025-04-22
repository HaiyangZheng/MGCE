#!/bin/bash
#SBATCH --account cvl
#SBATCH -p general
#SBATCH --qos normal
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=20000
#SBATCH --gres gpu:1
#SBATCH -o /home/psawl/hyzheng/debug_fedgcd/DPC2L/hyzheng/log_final/scars_fixnumworker_uon_test1.log

module load gcc/gcc-10.2.0
# module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0
module load nvidia/cuda-11.1 nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

source /home/psawl/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python 9.UON_explore_full.py \
 --dataset_name='scars' \
 --single_minsim_1=0.6 \
 --single_minsim_2=0.6 \
 --single_minsim_3=0.6 \
 --single_k1=15 \
 --k1_ratio=0.6 \
 --DCCL_contrastive_cluster_weight=0.1 \
 --align_loss_weight=0.1 \
 --seed=0 \
 --exp_name='scars_fixnumworker_uon_test1' \