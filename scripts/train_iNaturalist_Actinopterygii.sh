CONFIG_VALUES=$(python -c "
from config import exp_root, osr_split_dir, dino_pretrain_path, inaturalist_root
print(f'--exp_root={exp_root}')
print(f'--osr_split_dir={osr_split_dir}')
print(f'--dino_pretrain_path={dino_pretrain_path}')
print(f'--inaturalist_root={inaturalist_root}')
" | tr '\n' ' ')

CUDA_VISIBLE_DEVICES=0 python train.py \
 --dataset_name='Actinopterygii' \
 --single_minsim_1=0.6 \
 --single_minsim_2=0.6 \
 --single_minsim_3=0.6 \
 --single_k1=26 \
 --k1_ratio=0.6 \
 --DCCL_contrastive_cluster_weight=0.1 \
 --DCCL_align_weight=0.1 \
 --seed=0 \
 --lr=0.05 \
 --exp_name='iNaturalist_Actinopterygii_seed0' \
 --runner_name='MGCE' \
 --token_cache_dir='/DATA/temp/token_cache_Actinopterygii' \
 --feature_output_index=0 \
 --sup_weight=0.35 \
 $CONFIG_VALUES