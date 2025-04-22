import os
from copy import deepcopy
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from data.data_utils import subsample_instances, dirichlet_split_noniid
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]
    temp = deepcopy(dataset.targets)
    temp_target = np.array(temp).astype(np.int64)
    dataset.targets = temp_target[mask]

    return dataset


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


class NIH_CXR_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, open_img=default_loader, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.loader = open_img

        self.CLASSES = [
            'No Finding', 'Infiltration', 'Atelectasis', 'Effusion', 'Nodule',
            'Mass', 'Pneumothorax', 'Consolidation', 'Pleural_Thickening',
            'Cardiomegaly', 'Fibrosis', 'Edema', 'Tortuous Aorta', 'Emphysema',
            'Pneumonia', 'Calcification of the Aorta', 'Pneumoperitoneum', 'Hernia',
            'Subcutaneous Emphysema', 'Pneumomediastinum'
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'nih-cxr-lt_single-label_{split}.csv'))

        self.img_paths = self.label_df['id'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()
        self.data = np.array(sorted(self.img_paths))
        self.uq_idxs = np.array(range(len(self.img_paths)))
        self.targets = np.array(self.labels)
        self.transform = transform

        # if self.split == 'train':
        #     self.transform = torchvision.transforms.Compose([
        #         torchvision.transforms.ToPILImage(),
        #         torchvision.transforms.RandomHorizontalFlip(),
        #         torchvision.transforms.RandomRotation(15),
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
        #     ])
        # else:
        #     self.transform = torchvision.transforms.Compose([
        #         torchvision.transforms.ToPILImage(),
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
        #     ])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.loader(self.data[idx])
        target = self.targets[idx]
        uq_idxs = self.uq_idxs[idx]
        # x = cv2.imread(self.data[idx])
        # x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_AREA)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, uq_idxs

        # return img, target, self.uq_idxs[idx], attribute
        #
        # x = self.transform(x)



        # return torch.as_tensor(np.array(img).astype('float')), torch.from_numpy(target).long(), torch.from_numpy(uq_idxs).long()






# back up for hetero labeled diata 0.001
def get_hete_federated_cxr_datasets(train_transform, test_transform, federated_args,  train_classes=range(10), prop_train_labels=0.8,
                    split_train_val=False, seed=0):

    np.random.seed(seed)


    label_dir = os.path.join(federated_args.cxr_root, 'LongTailCXR')
    data_dir = os.path.join(federated_args.cxr_root, 'images')
    whole_training_set = NIH_CXR_Dataset(data_dir=data_dir, label_dir=label_dir, split='train', transform=train_transform)
    train_labels = np.array(deepcopy(whole_training_set.targets))

    client_idcs = dirichlet_split_noniid(train_labels=train_labels,
                                         alpha=federated_args.dirichlet,
                                         n_clients=federated_args.n_clients)

    #### visulization statistic ##############
    # Get labelled training set which has subsampled classes, then subsample some indices from that

    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # train_labels_set = set(whole_training_set.targets)
    # plt.figure(figsize=(20, 3))
    # plt.hist([train_labels[idc] for idc in client_idcs], stacked=True,
    #          bins=np.arange(min(train_labels) - 0.5, max(train_labels) + 1.5, 1),
    #          label=["Client {}".format(i) for i in range(federated_args.num_clients)], rwidth=0.5)
    # plt.xticks(np.arange(len(train_labels_set)), train_labels_set)
    # plt.legend()
    # plt.show()

    set_list = [set(train_labels[idc]) for idc in client_idcs]
    unit_set = set()
    # com_set = set()
    info_dict = {}
    for _i, __set in enumerate(set_list):
        if federated_args.show_client_data_info:
            print(f'The number of classes in the {_i}-th client: {len(__set)}')
            print(f'The classes in the {_i}-th client: {__set}')
            print(f'The number of images in the {_i}-th client: {len(client_idcs[_i])}')
        info_dict.update({f'client-{_i}': (len(__set),
                                           __set,
                                           len(client_idcs[_i]),
                                           client_idcs[_i])})
        unit_set = unit_set | __set
    if federated_args.show_client_data_info:
        print(f'The classes are totally include in across all clients: {unit_set}')

    info_dict.update({'total classes': unit_set})
    for _set in set_list:
        unit_set = _set & unit_set
    if federated_args.show_client_data_info:
        print(f'The classes shared across all clients: {unit_set}')

    info_dict.update({'shared classes': unit_set})
    federated_test_dataset = {}
    federated_train_dataset_labelled = {}
    federated_train_dataset_unlabelled = {}
    federated_train_dataset_val = {}

    for index_client in range(federated_args.n_clients):
        client_train_dataset = subsample_dataset(deepcopy(whole_training_set), client_idcs[index_client])
        client_train_labeled_classes = np.random.choice(np.unique(client_train_dataset.targets), len(set(client_train_dataset.targets))//2, replace=False)
        # client_train_labeled_classes = client_train_labeled_classes - 1 # align GCD only for cub
        train_dataset_labelled = subsample_classes(deepcopy(client_train_dataset), include_classes=client_train_labeled_classes)
        # train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
        # subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)

        # train_dataset_labelled = deepcopy(client_train_dataset)
        subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
        train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
        # Split into training and validation sets
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

        # Get unlabelled data
        unlabelled_indices_uq_idx = set(client_train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
        # unlabelled_indices_uq_idx = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
        uq_idx_2_array_index = {}
        unlabelled_indices_2_array_index = []
        for idx_ in unlabelled_indices_uq_idx:
            temp = np.where(idx_ == client_train_dataset.uq_idxs)[0]
            assert len(temp) == 1
            unlabelled_indices_2_array_index.append(temp[0])
        train_dataset_unlabelled = subsample_dataset(deepcopy(client_train_dataset),
                                                     np.array(unlabelled_indices_2_array_index))
        # Get test set for all classes
        test_dataset = NIH_CXR_Dataset(data_dir=data_dir, label_dir=label_dir, split='test')
        train_dataset_labelled_split = train_dataset_labelled_split if split_train_val else train_dataset_labelled
        federated_train_dataset_labelled[f'client-{index_client}'] = deepcopy(train_dataset_labelled_split)
        val_dataset_labelled_split = val_dataset_labelled_split if split_train_val else None
        federated_train_dataset_val[f'client-{index_client}'] = deepcopy(val_dataset_labelled_split)
        federated_test_dataset[f'client-{index_client}'] = deepcopy(test_dataset)
        federated_train_dataset_unlabelled[f'client-{index_client}'] = deepcopy(train_dataset_unlabelled)

    labeled_set_list = [federated_train_dataset_labelled[f'client-{index_client}'].targets.tolist() for index_client in range(federated_args.n_clients)]
    unlabeled_set_list = [federated_train_dataset_unlabelled[f'client-{index_client}'].targets.tolist() for index_client in range(federated_args.n_clients)]
    # labeled_set_list = [train_dataset_labelled.targets[i_client] for i_client in client_idcs_train_labeled]
    # unlabeled_set_list = [train_dataset_unlabelled.targets[i_client] for i_client in client_idcs_train_unlabeled]
    labeled_unit_set = set()
    unlabeled_unit_set = set()
    # com_set = set()
    info_dict = {}
    for _i, __set in enumerate(labeled_set_list):
        if federated_args.show_client_data_info:
            print(f'The number of labeled classes in the {_i}-th client: {len(set(__set))}')
            print(f'The number of unlabeled classes in the {_i}-th client: {len(set(unlabeled_set_list[_i]))}')
            print(f'The number of labeled images in the {_i}-th client: {len(labeled_set_list[_i])}')
            print(f'The number of unlabeled images in the {_i}-th client: {len(unlabeled_set_list[_i])}')
            print(
                f'The number of labeled and unlabeled classes in the {_i}-th client: {len(set(__set) | set(unlabeled_set_list[_i]))}')
        info_dict.update({f'client-{_i}': (len(set(__set)),
                                           len(set(unlabeled_set_list[_i])),
                                           len(labeled_set_list[_i]),
                                           len(unlabeled_set_list[_i]),
                                           len(set(__set) | set(unlabeled_set_list[_i])))})
        labeled_unit_set = labeled_unit_set | set(__set)
        unlabeled_unit_set = unlabeled_unit_set | set(unlabeled_set_list[_i])
    info_dict.update({'global seen classes': labeled_unit_set})
    info_dict.update({'num global classes': 20})
    if federated_args.show_client_data_info:
        print(f'The labeled {len(labeled_unit_set)} classes are totally include across all clients: {labeled_unit_set}')

        print(
            f'The unlabeled {len(unlabeled_unit_set)} classes are totally include across all clients: {unlabeled_unit_set}')
    count_dict = {}
    for i in list(labeled_unit_set):
        count_dict[i] = len([i for _list in labeled_set_list if i in _list])
    n = 0
    for v in count_dict.values():
        if v >= 2:
            n = n + 1
    if federated_args.show_client_data_info:
        print(f'shared across at least 2, <{n}> classes')
    info_dict.update({'total labeled classes': labeled_unit_set})
    info_dict.update({'total unlabeled classes': unlabeled_unit_set})
    for _set in labeled_set_list:
        labeled_joint_set = set(_set) & labeled_unit_set
        unlabeled_unit_set = set(_set) & unlabeled_unit_set
    if federated_args.show_client_data_info:
        print(f'The {len(labeled_joint_set)} labeled classes shared across all clients: {labeled_joint_set}')
        print(f'The {len(unlabeled_unit_set)} unlabeled classes shared across all clients: {unlabeled_unit_set}')

    info_dict.update({'labeled shared classes': labeled_joint_set})
    info_dict.update({'unlabeled shared classes': unlabeled_unit_set})


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
    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None
    # Get test set for all classes
    # test_dataset = CustomCub2011(root=federated_args.cub_root, transform=test_transform, train=False, download=False)
    whole_test_dataset = NIH_CXR_Dataset(data_dir=data_dir, label_dir=label_dir, split='test')

    test_dataset_labelled = subsample_classes(deepcopy(whole_test_dataset), include_classes=train_classes)
    test_subsample_indices = subsample_instances(test_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    test_dataset_labelled = subsample_dataset(test_dataset_labelled, test_subsample_indices)
    test_unlabelled_indices = set(whole_test_dataset.uq_idxs) - set(test_dataset_labelled.uq_idxs)
    test_dataset_unlabelled = subsample_dataset(deepcopy(whole_test_dataset), np.array(list(test_unlabelled_indices)))


    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    test_info_update_dict = {'number_of_classes_in_whole_train_set': len(list(train_classes)),
                             'number_of_classes_in_test_set_labeled': len(set(test_dataset_labelled.targets)),
                             'number_of_images_in_test_set_labeled': len(test_dataset_labelled.targets),
                             'number_of_classes_in_test_set_unlabeled': len(set(test_dataset_unlabelled.targets)),
                             'number_of_images_in_test_set_unlabeled': len(test_dataset_unlabelled.targets)}
    if federated_args.show_client_data_info:
        print(test_info_update_dict)
    info_dict.update(test_info_update_dict)
    whole_test_dataset.information = info_dict


    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': whole_test_dataset,
        'test_labelled': test_dataset_labelled,
        'test_unlabelled': test_dataset_unlabelled
    }
    for client_index, train_labelled in federated_train_dataset_labelled.items():
        all_datasets.update({f'federated_train_labelled_{client_index}': train_labelled})
        if federated_args.show_client_data_info:
            print(f'The number of labelled images in {client_index}: {len(train_labelled)}')
    for client_index, train_unlabelled in federated_train_dataset_unlabelled.items():
        all_datasets.update({f'federated_train_unlabelled_{client_index}': train_unlabelled})
        if federated_args.show_client_data_info:
            print(f'The number of unlabelled images in {client_index}: {len(train_unlabelled)}')

    for __i, (client_index, test_) in enumerate(federated_test_dataset.items()):
        all_datasets.update({f'federated_test_{client_index}': test_})
        if federated_args.show_client_data_info:
            print(f'The number of test images in {client_index}: {len(test_)}')
    for client_index, train_unlabelled in federated_train_dataset_val.items():
        all_datasets.update({f'federated_train_val_{client_index}': train_unlabelled})
        if train_unlabelled is not None:
            if federated_args.show_client_data_info:
                print(f'The number of val images in {client_index}: {len(train_unlabelled)}')
    return all_datasets
































class MIMIC_CXR_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split):
        self.split = split

        self.CLASSES = [
            'No Finding', 'Lung Opacity', 'Cardiomegaly', 'Atelectasis',
            'Pleural Effusion', 'Support Devices', 'Edema', 'Pneumonia',
            'Pneumothorax', 'Lung Lesion', 'Fracture', 'Enlarged Cardiomediastinum',
            'Consolidation', 'Pleural Other', 'Calcification of the Aorta',
            'Tortuous Aorta', 'Pneumoperitoneum', 'Subcutaneous Emphysema',
            'Pneumomediastinum'
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'mimic-lt_single-label_{split}.csv'))

        self.img_paths = self.label_df['path'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        self.labels = self.label_df[self.CLASSES].idxmax(axis=1).apply(lambda x: self.CLASSES.index(x)).values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])
        x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_AREA)
        x = self.transform(x)

        y = np.array(self.labels[idx])

        return x.float(), torch.from_numpy(y).long()

## CREDIT TO https://github.com/agaldran/balanced_mixup ##

# pytorch-wrapping-multi-dataloaders/blob/master/wrapping_multi_dataloaders.py
class ComboIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [loader_iter.next() for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)

class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches