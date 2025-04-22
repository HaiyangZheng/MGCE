import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

# def subsample_instances(dataset, prop_indices_to_subsample=0.8):

#     np.random.seed(0)
#     subsample_indices = []
#     index_per_class_list = [np.where(tar == np.array(dataset.targets))[0] for tar in list(set(dataset.targets))]
#     for per_class_index in index_per_class_list:
#         _selected = np.random.choice(per_class_index, replace=False,
#                          size=(int(prop_indices_to_subsample * len(per_class_index)),))
#         subsample_indices.append(_selected)
#     # subsample_indices = np.random.choice(range(len(dataset)), replace=False,
#     #                                      size=(int(prop_indices_to_subsample * len(dataset)),))
#     subsample_indices = np.concatenate(subsample_indices, axis=0)
#     return subsample_indices
def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

def subsample_instances_in_according_to_uq_idxs(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)

    subsample_indices = np.random.choice(dataset.uq_idxs, replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

import copy
from torchvision.transforms import Compose

class Add_Old_Class_Mask(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, row_datset, labeled_or_not, relabel_dict, transform):
        self.relabel_dict = relabel_dict
        self.dataset = copy.deepcopy(row_datset)
        self.old_class_mask = self.dataset.old_class_mask
        self.dataset.transform = transform
        if hasattr(self.dataset, "information"):
            self.information = self.dataset.information

        if labeled_or_not == 1 or labeled_or_not == 0:
            self.dataset.target_transform = None
            self.labeled_or_not = labeled_or_not
        else:
            assert False, "unexpected"


    def __getitem__(self, item):
        _tuple = self.dataset[item]
        img, label, uq_idx = _tuple[0],_tuple[1],_tuple[2]    
        return img, self.relabel_dict[label], uq_idx, np.array([self.labeled_or_not]), self.dataset.old_class_mask[item]
       
            
    def __len__(self):
       
        return len(self.dataset)
























class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset=None, re_label=False):
        _labelled_dataset = copy.deepcopy(labelled_dataset)
        _unlabelled_dataset = copy.deepcopy(unlabelled_dataset)

        self.labelled_dataset = _labelled_dataset
        self.unlabelled_dataset = _unlabelled_dataset

        self.target_transform = None

        if unlabelled_dataset is not None:
            if hasattr(labelled_dataset, 'data'):
                if isinstance(labelled_dataset.data, list):
                    self.data = _labelled_dataset.data + _unlabelled_dataset.data
                else:
                    self.data = np.concatenate((_labelled_dataset.data, _unlabelled_dataset.data), axis=0)
            elif hasattr(labelled_dataset, 'samples'):
                 self.data = _labelled_dataset.samples + _unlabelled_dataset.samples
            else:
                assert False, f'Unsuport {labelled_dataset}'
            if hasattr(_labelled_dataset, 'targets'):
                if isinstance(labelled_dataset.targets, list):
                    self.targets = _labelled_dataset.targets + _unlabelled_dataset.targets
                else:
                    self.targets = np.concatenate((_labelled_dataset.targets, _unlabelled_dataset.targets), axis=0)
            else:
                assert False, f'Unsuport {labelled_dataset}'
            if hasattr(_labelled_dataset, 'uq_idxs'):
                self.uq_idxs = _labelled_dataset.uq_idxs.tolist() + _unlabelled_dataset.uq_idxs.tolist()
            else:
                assert False, f'Unsuport {labelled_dataset}'
            
            if hasattr(_labelled_dataset, 'old_class_mask'):
                self.old_class_mask = _labelled_dataset.old_class_mask + _unlabelled_dataset.old_class_mask
            else:
                assert False, f'Unsuport {labelled_dataset}'

        if re_label:
            self.relabel_dict = {}
            for new_label, old_label in enumerate(np.unique(self.targets).tolist()):
                self.relabel_dict[old_label] = new_label

        else:
            self.relabel_dict = None
        print(f'Merge dataset -> relabel_dict:{self.relabel_dict}')


    def __getitem__(self, item):
        if self.unlabelled_dataset is None:
            _tuple = self.unlabelled_dataset[item]
            img, label, uq_idx = _tuple[0],_tuple[1],_tuple[2]
            labeled_or_not = 0 # for unlabel_train_sample_test
            if self.relabel_dict is None:
                return img, label, uq_idx, np.array([labeled_or_not]), np.array([0])
            else:
                return img, self.relabel_dict[label], uq_idx, np.array([labeled_or_not]), self.unlabelled_dataset.old_class_mask[item]
        else:
            if item < len(self.labelled_dataset):
                _tuple = self.labelled_dataset[item]
              
                img, label, uq_idx = _tuple[0],_tuple[1],_tuple[2]
                labeled_or_not = 1
                if self.relabel_dict is None:
                    return img, label, uq_idx, np.array([labeled_or_not]), np.array([self.old_class_mask[item]])
                else:
                    if label == -1 or item == -1:
                        print('find')
                    return img, self.relabel_dict[label], uq_idx, np.array([labeled_or_not]),  np.array([self.old_class_mask[item]])
            else:
                _tuple = self.unlabelled_dataset[item - len(self.labelled_dataset)]
                img, label, uq_idx = _tuple[0], _tuple[1], _tuple[2]
                labeled_or_not = 0
                if self.relabel_dict is None:
                    return img, label, uq_idx, np.array([labeled_or_not]), np.array([0])
                else:
                    return img, self.relabel_dict[label], uq_idx, np.array([labeled_or_not]),  np.array([self.old_class_mask[item]])
    def __len__(self):
        if self.unlabelled_dataset is None:
            return len(self.labelled_dataset)
        else:

            return len(self.unlabelled_dataset) + len(self.labelled_dataset)

