from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np

from data.data_utils import subsample_instances
# from config import cifar_10_root, cifar_100_root
from PIL import Image


class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):

        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


# class CustomCIFAR100(CIFAR100):

#     def __init__(self, *args, **kwargs):
#         super(CustomCIFAR100, self).__init__(*args, **kwargs)

#         self.uq_idxs = np.array(range(len(self)))

#     def __getitem__(self, item):
#         img, label = super().__getitem__(item)
#         uq_idx = self.uq_idxs[item]

#         return img, label, uq_idx

#     def __len__(self):
#         return len(self.targets)


class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, sample_subset=False, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)
        
        self.sample_subset = sample_subset
        
        if self.sample_subset:
            self._create_subset()
        else:
            self.uq_idxs = np.array(range(len(self.targets)))
    
    def _create_subset(self):
        """从每个类别随机采样1/10的数据"""
        # 获取所有标签
        targets = np.array(self.targets)
        
        # 存储采样后的索引
        sampled_indices = []
        
        # 对每个类别进行采样
        for class_id in range(100):  # CIFAR-100有100个类别
            # 找到当前类别的所有索引
            class_indices = np.where(targets == class_id)[0]
            
            # 计算需要采样的数量（1/10）
            sample_size = max(1, len(class_indices) // 10)  # 至少采样1个
            
            # 随机采样
            if len(class_indices) > 0:
                sampled_class_indices = np.random.choice(
                    class_indices, 
                    size=sample_size, 
                    replace=False
                )
                sampled_indices.extend(sampled_class_indices)
        
        # 排序采样的索引
        sampled_indices = sorted(sampled_indices)
        
        # 更新数据集
        self.data = self.data[sampled_indices]
        # 更新标签
        self.targets = [self.targets[i] for i in sampled_indices]
        
        # 创建新的唯一索引
        self.uq_idxs = np.array(range(len(self.targets)))
        
        # 保存原始索引映射（可选，用于追踪原始数据）
        self.original_indices = np.array(sampled_indices)

    def __getitem__(self, item):
        if self.sample_subset:
            # 对于采样的数据集，直接从已更新的data和targets获取
            img, target = self.data[item], self.targets[item]
            
            # 转换为PIL图像
            img = Image.fromarray(img)
            
            # 应用变换
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
                
            uq_idx = self.uq_idxs[item]
            return img, target, uq_idx
        else:
            # 使用原始方法
            img, label = super().__getitem__(item)
            uq_idx = self.uq_idxs[item]
            return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_cifar_10_datasets(train_transform, test_transform, train_classes=(0, 1, 8, 9),
                       prop_train_labels=0.8, split_train_val=False, seed=0, args=None):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCIFAR10(root=args.cifar_10_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CustomCIFAR10(root=args.cifar_10_root, transform=test_transform, train=False)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


def get_cifar_100_datasets(train_transform, test_transform, train_classes=range(80),
                       prop_train_labels=0.8, split_train_val=False, seed=0, args=None):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCIFAR100(root=args.cifar_100_root, sample_subset=args.cifar100_sample_subset, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = CustomCIFAR100(root=args.cifar_100_root, sample_subset=args.cifar100_sample_subset, transform=test_transform, train=False)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


if __name__ == '__main__':

    args = type('', (), {})()  # Create a dummy args object
    args.cifar_100_root = '/leonardo_work/IscrC_DiffGRE/Datasets/cifar100'
    args.cifar100_sample_subset = True

    x = get_cifar_100_datasets(None, None, split_train_val=False,
                         train_classes=range(80), prop_train_labels=0.5, args=args)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')