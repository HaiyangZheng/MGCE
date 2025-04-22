#!/bin/bash
#SBATCH -A IscrC_HDSCisLa
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --time 1-00:00:00     # format: HH:MM:SS
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=100000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=tar_inaturalist

module load cuda/12.1
source /leonardo/home/userexternal/hzheng00/miniconda3/bin/activate fedgcd

CUDA_VISIBLE_DEVICES=0 python inaturalist_tar.py