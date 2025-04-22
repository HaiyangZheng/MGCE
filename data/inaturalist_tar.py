import tarfile
import os

def extract_tar_gz(filepath, extract_to):
    """
    解压 .tar.gz 文件到指定目录。
    :param filepath: .tar.gz 文件的路径。
    :param extract_to: 解压目录。
    """
    try:
        # 确保解压目录存在
        os.makedirs(extract_to, exist_ok=True)
        
        # 打开 .tar.gz 文件
        with tarfile.open(filepath, "r:gz") as tar:
            # 提取所有内容
            tar.extractall(path=extract_to)
    except Exception as e:
        print(f"解压过程中发生错误：{e}")

# 指定 .tar.gz 文件的路径
tar_gz_path = '/leonardo_work/IscrC_Fed-GCD/GCD_datasets/iNaturalist/test2017.tar.gz'
# 指定解压目录
extract_to_path = '/leonardo_work/IscrC_Fed-GCD/GCD_datasets/iNaturalist'

# 调用函数开始解压
extract_tar_gz(tar_gz_path, extract_to_path)