import torchvision
import numpy as np

import os
from data.data_utils import subsample_instances, subsample_instances_in_according_to_uq_idxs, dirichlet_split_noniid
from copy import deepcopy



class ImageNetBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(ImageNetBase, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx


def subsample_dataset(dataset, idxs):

    imgs_ = []
    for i in idxs:
        imgs_.append(dataset.imgs[i])
    dataset.imgs = imgs_

    samples_ = []
    for i in idxs:
        samples_.append(dataset.samples[i])
    dataset.samples = samples_

    # dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    # dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]

    # dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.targets = np.array(dataset.targets)[idxs]
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=list(range(1000))):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2


def get_imagenet_100_datasets(train_transform, test_transform, federated_args,  train_classes=range(80),
                           prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)

    # Subsample imagenet dataset initially to include 100 classes
    subsampled_100_classes = np.random.choice(range(1000), size=(100,), replace=False)
    subsampled_100_classes = np.sort(subsampled_100_classes)
    print(f'Constructing ImageNet-100 dataset from the following classes: {subsampled_100_classes.tolist()}')
    cls_map = {i: j for i, j in zip(subsampled_100_classes, range(100))}

    # Init entire training set
    imagenet_training_set = ImageNetBase(root=os.path.join(federated_args.imagenet_root, 'train'), transform=train_transform)
    whole_training_set = subsample_classes(imagenet_training_set, include_classes=subsampled_100_classes)

    # Reset dataset
    whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    whole_training_set.target_transform = None

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
    test_dataset = ImageNetBase(root=os.path.join(federated_args.imagenet_root, 'val'), transform=test_transform)
    test_dataset = subsample_classes(test_dataset, include_classes=subsampled_100_classes)

    # Reset test set
    test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    test_dataset.target_transform = None

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    # all_datasets = {
    #     'train_labelled': train_dataset_labelled,
    #     'train_unlabelled': train_dataset_unlabelled,
    #     'val': val_dataset_labelled,
    #     'test': test_dataset,
    # }
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'test_labelled': test_dataset,
        'test_unlabelled': test_dataset
    }

    return all_datasets


def get_hete_federated_imagenet_100_datasets(train_transform, test_transform, federated_args,  train_classes=range(80),
                           prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)

    # Subsample imagenet dataset initially to include 100 classes
    subsampled_100_classes = np.random.choice(range(1000), size=(100,), replace=False)
    subsampled_100_classes = np.sort(subsampled_100_classes)
    print(f'Constructing ImageNet-100 dataset from the following classes: {subsampled_100_classes.tolist()}')
    cls_map = {i: j for i, j in zip(subsampled_100_classes, range(100))}

    # Init entire training set
    imagenet_training_set = ImageNetBase(root=os.path.join(federated_args.imagenet_root, 'train'), transform=train_transform)
    whole_training_set = subsample_classes(imagenet_training_set, include_classes=subsampled_100_classes)

    # Reset dataset
    whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    whole_training_set.target_transform = None


    # Get test set for all classes
    whole_test_dataset = ImageNetBase(root=os.path.join(federated_args.imagenet_root, 'val'), transform=test_transform)
    whole_test_dataset = subsample_classes(whole_test_dataset, include_classes=subsampled_100_classes)

    # Reset test set
    whole_test_dataset.samples = [(s[0], cls_map[s[1]]) for s in whole_test_dataset.samples]
    whole_test_dataset.targets = [s[1] for s in whole_test_dataset.samples]
    whole_test_dataset.uq_idxs = np.array(range(len(whole_test_dataset)))
    whole_test_dataset.target_transform = None

    # whole_training_set = CustomCIFAR10(root=federated_args.cifar_10_root, transform=train_transform, train=True)

    # train_labels = np.array(deepcopy(whole_training_set.targets))
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
    # val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    client_idcs_train_labeled = dirichlet_split_noniid(train_labels=train_dataset_labelled.targets,
                                                       alpha=federated_args.dirichlet,
                                                       n_clients=federated_args.n_clients)

    client_idcs_train_unlabeled = dirichlet_split_noniid(train_labels=train_dataset_unlabelled.targets,
                                                         alpha=federated_args.dirichlet,
                                                         n_clients=federated_args.n_clients)

    labeled_set_list = [train_dataset_labelled.targets[i_client] for i_client in client_idcs_train_labeled]
    unlabeled_set_list = [train_dataset_unlabelled.targets[i_client] for i_client in client_idcs_train_unlabeled]
    labeled_unit_set = set()
    unlabeled_unit_set = set()
    # com_set = set()
    info_dict = {}
    for _i, __set in enumerate(labeled_set_list):
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
    info_dict.update({'num global classes': 100})
    print(f'The labeled {len(labeled_unit_set)} classes are totally include across all clients: {labeled_unit_set}')
    print(
        f'The unlabeled {len(unlabeled_unit_set)} classes are totally include across all clients: {unlabeled_unit_set}')

    info_dict.update({'total labeled classes': labeled_unit_set})
    info_dict.update({'total unlabeled classes': unlabeled_unit_set})
    count_dict = {}
    for i in list(labeled_unit_set):
        count_dict[i] = len([i for _list in labeled_set_list if i in _list])
    n = 0
    for v in count_dict.values():
        if v >= 2:
            n = n + 1
    print(f'shared across at least 2, <{n}> classes')
    n = 0
    for v in count_dict.values():
        if v == 2:
            n = n + 1
    print(f'shared across = 2, <{n}> classes')
    for _set in labeled_set_list:
        labeled_unit_set = set(_set) & labeled_unit_set
        unlabeled_unit_set = set(_set) & unlabeled_unit_set
    print(f'The {len(labeled_unit_set)} labeled classes shared across all clients: {labeled_unit_set}')
    print(f'The {len(unlabeled_unit_set)} unlabeled classes shared across all clients: {unlabeled_unit_set}')

    info_dict.update({'labeled shared classes': labeled_unit_set})
    info_dict.update({'unlabeled shared classes': unlabeled_unit_set})

    federated_test_dataset = {}
    federated_train_dataset_labelled = {}
    federated_train_dataset_unlabelled = {}
    federated_train_dataset_val = {}

    for index_client in range(federated_args.n_clients):
        train_dataset_labelled = subsample_dataset(deepcopy(whole_training_set),
                                                   client_idcs_train_labeled[index_client])

        # subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
        # train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
        # Split into training and validation sets
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

        # Get unlabelled data

        train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set),
                                                     client_idcs_train_unlabeled[index_client])
        # Get test set for all classes
        test_dataset = deepcopy(whole_test_dataset)
        train_dataset_labelled_split = train_dataset_labelled_split if split_train_val else train_dataset_labelled
        federated_train_dataset_labelled[f'client-{index_client}'] = deepcopy(train_dataset_labelled_split)
        val_dataset_labelled_split = val_dataset_labelled_split if split_train_val else None
        federated_train_dataset_val[f'client-{index_client}'] = deepcopy(val_dataset_labelled_split)
        federated_test_dataset[f'client-{index_client}'] = deepcopy(test_dataset)
        federated_train_dataset_unlabelled[f'client-{index_client}'] = deepcopy(train_dataset_unlabelled)

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


    test_dataset_labelled = subsample_classes(deepcopy(whole_test_dataset), include_classes=train_classes)
    test_subsample_indices = subsample_instances(test_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    test_dataset_labelled = subsample_dataset(test_dataset_labelled, test_subsample_indices)
    test_unlabelled_indices = set(whole_test_dataset.uq_idxs) - set(test_dataset_labelled.uq_idxs)
    test_dataset_unlabelled = subsample_dataset(deepcopy(whole_test_dataset), np.array(list(test_unlabelled_indices)))
    test_info_update_dict = {'number_of_classes_in_whole_train_set': len(list(train_classes)),
                             'number_of_classes_in_test_set_labeled': len(set(test_dataset_labelled.targets)),
                             'number_of_images_in_test_set_labeled': len(test_dataset_labelled.targets),
                             'number_of_classes_in_test_set_unlabeled': len(set(test_dataset_unlabelled.targets)),
                             'number_of_images_in_test_set_unlabeled': len(test_dataset_unlabelled.targets)}
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
        print(f'The number of labelled images in {client_index}: {len(train_labelled)}')
    for client_index, train_unlabelled in federated_train_dataset_unlabelled.items():
        all_datasets.update({f'federated_train_unlabelled_{client_index}': train_unlabelled})
        print(f'The number of unlabelled images in {client_index}: {len(train_unlabelled)}')

    for __i, (client_index, test_) in enumerate(federated_test_dataset.items()):
        all_datasets.update({f'federated_test_{client_index}': test_})
        print(f'The number of test images in {client_index}: {len(test_)}')
    for client_index, train_unlabelled in federated_train_dataset_val.items():
        all_datasets.update({f'federated_train_val_{client_index}': train_unlabelled})
        if train_unlabelled is not None:
            print(f'The number of val images in {client_index}: {len(train_unlabelled)}')
    return all_datasets





if __name__ == '__main__':

    x = get_imagenet_100_datasets(None, None, split_train_val=False,
                               train_classes=range(50), prop_train_labels=0.5)

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