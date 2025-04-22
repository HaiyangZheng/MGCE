import os

import torchvision
import numpy as np
from copy import deepcopy

from config import herbarium_dataroot
from data.data_utils import subsample_instances, dirichlet_split_noniid


class HerbariumDataset19(torchvision.datasets.ImageFolder):

    def __init__(self, *args, **kwargs):

        # Process metadata json for training images into a DataFrame
        super().__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, idx):

        img, label = super().__getitem__(idx)
        uq_idx = self.uq_idxs[idx]

        return img, label, uq_idx


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()

    dataset.uq_idxs = dataset.uq_idxs[mask]

    dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    dataset.targets = [int(x) for x in dataset.targets]

    return dataset


def subsample_classes(dataset, include_classes=range(250)):

    cls_idxs = [x for x, l in enumerate(dataset.targets) if l in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_instances_per_class=5):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        # Have a balanced test set
        # v_ = np.random.choice(cls_idxs, replace=False, size=(val_instances_per_class,))
        v_ = np.random.choice(cls_idxs, replace=True, size=(val_instances_per_class,)) # without using val
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_herbarium_datasets(train_transform, test_transform, train_classes=range(500), prop_train_labels=0.8,
                            seed=0, split_train_val=False):

    np.random.seed(seed)

    # Init entire training set
    train_dataset = HerbariumDataset19(transform=train_transform,
                                            root=os.path.join(herbarium_dataroot, 'small-train'))

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    # TODO: Subsampling unlabelled set in uniform random fashion from training data, will contain many instances of dominant class
    train_dataset_labelled = subsample_classes(deepcopy(train_dataset), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    if split_train_val:

        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled,
                                                     val_instances_per_class=5)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

    else:

        train_dataset_labelled_split, val_dataset_labelled_split = None, None

    # Get unlabelled data
    unlabelled_indices = set(train_dataset.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset), np.array(list(unlabelled_indices)))

    # Get test dataset
    test_dataset = HerbariumDataset19(transform=test_transform,
                                            root=os.path.join(herbarium_dataroot, 'small-validation'))

    # Transform dict
    unlabelled_classes = list(set(train_dataset.targets) - set(train_classes))
    target_xform_dict = {}
    for i, k in enumerate(list(train_classes) + unlabelled_classes):
        target_xform_dict[k] = i

    test_dataset.target_transform = lambda x: target_xform_dict[x]
    train_dataset_unlabelled.target_transform = lambda x: target_xform_dict[x]

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
def get_hete_federated_herb_datasets(train_transform, test_transform, federated_args,  train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = HerbariumDataset19(transform=train_transform,
                                            root=os.path.join(federated_args.herbarium_dataroot, 'small-train'))

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
        # client_train_labeled_classes = client_train_labeled_classes - 1# align GCD only for cub
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
        test_dataset = HerbariumDataset19(transform=test_transform,
                                          root=os.path.join(federated_args.herbarium_dataroot, 'small-validation'))
        train_dataset_labelled_split = train_dataset_labelled_split if split_train_val else train_dataset_labelled
        federated_train_dataset_labelled[f'client-{index_client}'] = deepcopy(train_dataset_labelled_split)
        val_dataset_labelled_split = val_dataset_labelled_split if split_train_val else None
        federated_train_dataset_val[f'client-{index_client}'] = deepcopy(val_dataset_labelled_split)
        federated_test_dataset[f'client-{index_client}'] = deepcopy(test_dataset)
        federated_train_dataset_unlabelled[f'client-{index_client}'] = deepcopy(train_dataset_unlabelled)

    labeled_set_list = [federated_train_dataset_labelled[f'client-{index_client}'].targets for index_client in range(federated_args.n_clients)]
    unlabeled_set_list = [federated_train_dataset_unlabelled[f'client-{index_client}'].targets for index_client in range(federated_args.n_clients)]
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
    info_dict.update({'num global classes': 683})
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
    whole_test_dataset = HerbariumDataset19(transform=test_transform,
                                            root=os.path.join(federated_args.herbarium_dataroot, 'small-validation'))

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

if __name__ == '__main__':

    np.random.seed(0)
    train_classes = np.random.choice(range(683,), size=(int(683 / 2)), replace=False)

    x = get_herbarium_datasets(None, None, train_classes=train_classes,
                               prop_train_labels=0.5)

    assert set(x['train_unlabelled'].targets) == set(range(683))

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))
    print('Printing number of labelled classes...')
    print(len(set(x['train_labelled'].targets)))
    print('Printing total number of classes...')
    print(len(set(x['train_unlabelled'].targets)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')