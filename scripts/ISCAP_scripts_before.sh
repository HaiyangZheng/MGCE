#!/bin/bash
#SBATCH -A IscrC_Fed-GCD
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --time 1-00:00:00     # format: HH:MM:SS
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=100000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=dccl_pseudo_test
#SBATCH -o /leonardo_work/IscrC_Fed-GCD/hyzheng/debug_fedgcd/DPC2L/hyzheng/log_full/scars_dccl_threehead_k1(15-0.6)_alignloss(0.1)_dcclloss(0.3)_before.log

module load cuda/12.1
source /leonardo/home/userexternal/hzheng00/miniconda3/bin/activate fedgcd

CUDA_VISIBLE_DEVICES=0 python 5.ISCAP_DCCL_alignloss_base_multilevel_tosimgcd_otherdatasets_before.py \
 --dataset_name='scars' \
 --single_minsim_1=0.6 \
 --single_minsim_2=0.6 \
 --single_minsim_3=0.6 \
 --single_k1_1=15 \
 --single_k1_2=9 \
 --single_k1_3=25 \
 --DCCL_contrastive_cluster_weight=0.3 \
 --exp_name='scars_dccl_threehead_k1(15-0.6)_alignloss(0.1)_dcclloss(0.3)_before' \