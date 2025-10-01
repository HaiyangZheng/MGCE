CONFIG_VALUES=$(python -c "
from config import osr_split_dir, dino_pretrain_path, cifar_100_root
print(f'--osr_split_dir={osr_split_dir}')
print(f'--dino_pretrain_path={dino_pretrain_path}')
print(f'--cifar_100_root={cifar_100_root}')
" | tr '\n' ' ')

# Run KNN parameter estimation with your actual script
CUDA_VISIBLE_DEVICES=0 python get_knn.py \
    --dataset_name='cifar100' \
    --minsim=0.5 \
    --token_cache_dir='/DATA/temp/token_cache_cifar100' \
    --output_dir='log_knn/' \
    --exp_name='knn_search_cifar100' \
    $CONFIG_VALUES