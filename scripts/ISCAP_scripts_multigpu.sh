#!/bin/bash
#SBATCH -A IscrC_Fed-GCD
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH -N 1                    # 1 node
#SBATCH --ntasks-per-node=4     # 3 tasks
#SBATCH --time 4-00:00:00       # format: HH:MM:SS
#SBATCH --gres=gpu:4            # 4 GPUs
#SBATCH --mem=100000            # memory per node
#SBATCH --job-name=cifar100k1
#SBATCH -o /leonardo_work/IscrC_Fed-GCD/hyzheng/debug_fedgcd/DPC2L/hyzheng/log_explore/%x_%j.log

module load cuda/12.1
source /leonardo/home/userexternal/hzheng00/miniconda3/bin/activate fedgcd

k1_values=(150 200 250 300)  # array of k1 values
for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$i python 5.ISCAP_DCCL_alignloss_base_multilevel_tosimgcd_otherdatasets.py \
     --dataset_name='cifar100' \
     --single_minsim_1=0.5 \
     --single_minsim_2=0.5 \
     --single_minsim_3=0.5 \
     --single_k1=${k1_values[$i]} \
     --k1_ratio=0.6 \
     --DCCL_contrastive_cluster_weight=0.1 \
     --align_loss_weight=0.1 \
     --exp_name="cifar100_dccl_threehead_k1(${k1_values[$i]}-0.6)_alignloss(0.1)_dcclloss(0.1)_soft" \
     > /leonardo_work/IscrC_Fed-GCD/hyzheng/debug_fedgcd/DPC2L/hyzheng/log_explore/cifar100_k1_${k1_values[$i]}.log 2>&1 &
done
wait