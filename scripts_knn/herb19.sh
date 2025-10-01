CONFIG_VALUES=$(python -c "
from config import osr_split_dir, dino_pretrain_path, herbarium_dataroot
print(f'--osr_split_dir={osr_split_dir}')
print(f'--dino_pretrain_path={dino_pretrain_path}')
print(f'--herbarium_dataroot={herbarium_dataroot}')
" | tr '\n' ' ')

# Run KNN parameter estimation with your actual script
CUDA_VISIBLE_DEVICES=0 python get_knn.py \
    --dataset_name='herbarium_19' \
    --minsim=0.6 \
    --token_cache_dir='/DATA/temp/token_cache_herb19' \
    --output_dir='log_knn/' \
    --exp_name='knn_search_herb19' \
    $CONFIG_VALUES