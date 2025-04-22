#!/bin/bash
#SBATCH -A IscrC_HDSCisLa
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --time 1-00:00:00     # format: HH:MM:SS
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=100000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=swoalign
#SBATCH -o /leonardo_work/IscrC_Fed-GCD/hyzheng/debug_fedgcd/DPC2L/hyzheng/log_paper/scars_woalignloss_seed(0).log

module load cuda/12.1
source /leonardo/home/userexternal/hzheng00/miniconda3/bin/activate fedgcd

CUDA_VISIBLE_DEVICES=0 python 15.ISCAP_full.py \
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
 --exp_name='scars_woalignloss_seed(0)' \