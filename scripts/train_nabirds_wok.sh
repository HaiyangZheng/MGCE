CONFIG_VALUES=$(python -c "
from config import exp_root, osr_split_dir, dino_pretrain_path, nabirds_root
print(f'--exp_root={exp_root}')
print(f'--osr_split_dir={osr_split_dir}')
print(f'--dino_pretrain_path={dino_pretrain_path}')
print(f'--nabirds_root={nabirds_root}')
" | tr '\n' ' ')

CUDA_VISIBLE_DEVICES=0 python train_wok.py \
 --dataset_name='nabirds' \
 --single_minsim_1=0.6 \
 --single_minsim_2=0.6 \
 --single_minsim_3=0.6 \
 --single_k1=10 \
 --k1_ratio=0.6 \
 --DCCL_contrastive_cluster_weight=0.1 \
 --DCCL_align_weight=0.1 \
 --seed=0 \
 --lr=0.05 \
 --dccl_update_freq=1 \
 --exp_name='Wok_Nabirds_seed0' \
 --runner_name='MGCE' \
 --token_cache_dir='/DATA/temp/token_cache_nabirds' \
 --feature_output_index=0 \
 --sup_weight=0.35 \
 $CONFIG_VALUES
# 0 for projector, 2 for backbone