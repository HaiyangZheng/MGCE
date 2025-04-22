import os
import os.path
import pathlib
from typing import Any, Callable, Optional, Union, Tuple
from typing import Sequence
from data.data_utils import subsample_instances, subsample_instances_in_according_to_uq_idxs, dirichlet_split_noniid

import numpy as np
from copy import deepcopy
from scipy import io as mat_io

from data.data_utils import subsample_instances
# from config import pets_root

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances


def make_dataset(dir, image_ids, targets):
    assert (len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


class OxfordIIITPet(Dataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

                - ``category`` (int): Label for one of the 37 pet categories.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.

            If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    valid_target_types = ("category", "segmentation")
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(
            self,
            root: str,
            split: str = "trainval",
            target_types: Union[Sequence[str], str] = "category",
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            loader=default_loader
    ):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        self.split = split

        if isinstance(target_types, str):
            target_types = [target_types]

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self._base_folder = pathlib.Path(self.root)
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        image_ids = []
        self._labels = []
        with open(self._anns_folder / f"{self.split}.txt") as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        samples = make_dataset(self.root, image_ids, self._labels)
        self.samples = samples
        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]

        self.uq_idxs = np.array(range(len(self)))

        self.data = [(uq_id, f_path, target) for uq_id, (f_path, target) in zip(self.uq_idxs, samples)]
        self.targets = [target for uq_id, (f_path, target) in zip(self.uq_idxs, samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.uq_idxs[idx]

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)


def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = [(p, t) for i, (p, t) in enumerate(dataset.samples) if i in idxs]
    dataset.uq_idxs = dataset.uq_idxs[mask]
    dataset.targets = np.array(dataset.targets)[mask]


    return dataset


def subsample_classes(dataset, include_classes=range(60)):
    cls_idxs = [i for i, (p, t) in enumerate(dataset.samples) if t in include_classes]  # 1885

    # TODO: Don't transform targets for now
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):
    all_targets = [t for i, (p, t) in enumerate(train_dataset.samples)]
    train_classes = np.unique(all_targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(all_targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_pets_datasets(train_transform, test_transform, federated_args, train_classes=range(19), prop_train_labels=0.8,
                      split_train_val=False, seed=0):
    np.random.seed(seed)

    # Init entire training set
    whole_training_set = OxfordIIITPet(root=federated_args.pets_root, transform=train_transform, split='trainval',
                                       download=False)  # len = 3680

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
    test_dataset = OxfordIIITPet(root=federated_args.pets_root, transform=test_transform, split='test', download=False)

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





# back up for hetero labeled diata 0.001
def get_hete_federated_pets_datasets(train_transform, test_transform, federated_args,  train_classes=range(19), prop_train_labels=0.8,
                    split_train_val=False, seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = OxfordIIITPet(root=federated_args.pets_root,
                                       transform=train_transform,
                                       split='trainval',
                                       download=False)
    train_labels = np.array(deepcopy(whole_training_set.targets))
    # train_labels = np.array(deepcopy(whole_training_set.targets)) - 1 only for cub
    # 1-200
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
        # client_train_dataset.targets = np.array(client_train_dataset.targets) - 1 # align cub  GCD
        client_train_dataset.targets = np.array(client_train_dataset.targets) # align GCD
        client_train_labeled_classes = np.random.choice(np.unique(client_train_dataset.targets), len(set(client_train_dataset.targets))//2, replace=False)
        # client_train_labeled_classes = client_train_labeled_classes - 1 # align GCD
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

        # train_dataset_unlabelled.client_train_labeled_classes = client_train_labeled_classes
        # train_dataset_labelled
        # Get test set for all classes
        test_dataset = OxfordIIITPet(root=federated_args.pets_root, transform=test_transform, split='test', download=False)
        test_dataset.targets = np.array(test_dataset.targets) - 1

        train_dataset_labelled_split = train_dataset_labelled_split if split_train_val else train_dataset_labelled
        federated_train_dataset_labelled[f'client-{index_client}'] = deepcopy(train_dataset_labelled_split)
        federated_train_dataset_labelled[f'client-{index_client}'].client_train_labeled_classes = client_train_labeled_classes
        val_dataset_labelled_split = val_dataset_labelled_split if split_train_val else None
        federated_train_dataset_val[f'client-{index_client}'] = deepcopy(val_dataset_labelled_split)
        federated_test_dataset[f'client-{index_client}'] = deepcopy(test_dataset)
        federated_test_dataset[f'client-{index_client}'].client_train_labeled_classes = client_train_labeled_classes
        federated_train_dataset_unlabelled[f'client-{index_client}'] = deepcopy(train_dataset_unlabelled)
        federated_train_dataset_unlabelled[f'client-{index_client}'].client_train_labeled_classes = client_train_labeled_classes

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
    info_dict.update({'num global classes': 200})
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
    whole_test_dataset = OxfordIIITPet(root=federated_args.pets_root, transform=test_transform, split="test",
                                       download=False)

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











def get_hete_federated_pets_datasets2(train_transform, test_transform, federated_args, train_classes=range(19),
                                     prop_train_labels=0.8,
                                     split_train_val=False, seed=0):
    np.random.seed(seed)

    # Init entire training set
    whole_training_set = OxfordIIITPet(root=federated_args.pets_root, transform=train_transform, split='trainval',
                                       download=False)  # len = 3680

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

    # Get test set for all classes
    # test_dataset = CustomCub2011(root=cub_root, transform=test_transform, train=False, download=False)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    # val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    client_idcs_train_labeled = dirichlet_split_noniid(train_labels=train_dataset_labelled.targets,
                                                       alpha=federated_args.dirichlet,
                                                       n_clients=federated_args.num_clients)

    client_idcs_train_unlabeled = dirichlet_split_noniid(train_labels=train_dataset_unlabelled.targets,
                                                         alpha=federated_args.dirichlet,
                                                         n_clients=federated_args.num_clients)

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
    info_dict.update({'num global classes': 36})
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
    for _set in labeled_set_list:
        labeled_joint_set = set(_set) & labeled_unit_set
        unlabeled_unit_set = set(_set) & unlabeled_unit_set
    print(f'The {len(labeled_joint_set)} labeled classes shared across all clients: {labeled_joint_set}')
    print(f'The {len(unlabeled_unit_set)} unlabeled classes shared across all clients: {unlabeled_unit_set}')

    info_dict.update({'labeled shared classes': labeled_joint_set})
    info_dict.update({'unlabeled shared classes': unlabeled_unit_set})

    federated_test_dataset = {}
    federated_train_dataset_labelled = {}
    federated_train_dataset_unlabelled = {}
    federated_train_dataset_val = {}

    for index_client in range(federated_args.num_clients):
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
        test_dataset = OxfordIIITPet(root=federated_args.pets_root, transform=test_transform, split='test',
                                     download=False)

        train_dataset_labelled_split = train_dataset_labelled_split if split_train_val else train_dataset_labelled
        federated_train_dataset_labelled[f'client-{index_client}'] = deepcopy(train_dataset_labelled_split)
        val_dataset_labelled_split = val_dataset_labelled_split if split_train_val else None
        federated_train_dataset_val[f'client-{index_client}'] = deepcopy(val_dataset_labelled_split)
        federated_test_dataset[f'client-{index_client}'] = deepcopy(test_dataset)
        federated_train_dataset_unlabelled[f'client-{index_client}'] = deepcopy(train_dataset_unlabelled)

    # labeled_class_set_each_client_list = [federated_train_dataset_labelled[f'client-{index_client}'].targets.tolist() for index_client in range(federated_args.num_clients)]

    # unit_set = set()
    # for _i, __set in enumerate(labeled_class_set_each_client_list):
    #     print(f'The number of classes in labeled data of the {_i}-th client: {len(set(__set))}')
    #     print(f'The classes in labeled data of the {_i}-th client: {set(__set)}')
    #     print(f'The number of labeled images in the {_i}-th client: {__set}')
    #     info_dict.update({f'labelled_client-{_i}': (len(set(__set)),
    #                                        set(__set),
    #                                        len(__set),
    #                                        __set)})
    #     unit_set = unit_set | set(__set)
    # print(f'The labeled classes are totally include in across all clients: {unit_set}')
    #
    # info_dict.update({'total labeled classes': unit_set})
    # for _set in labeled_class_set_each_client_list:
    #     unit_set = set(_set) & set(unit_set)
    # print(f'The labeled classes shared across all clients: {unit_set}')
    #
    # info_dict.update({'labeled shared classes': unit_set})

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
    whole_test_dataset = OxfordIIITPet(root=federated_args.pets_root, transform=test_transform, split='test',
                                       download=False)

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


def main():
    x = get_pets_datasets(None, None, split_train_val=False)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))
    print('Printing number of labelled classes...')
    print(len(set([i[1] for i in x['train_labelled'].samples])))
    print('Printing total number of classes...')
    print(len(set([i[1] for i in x['train_unlabelled'].samples])))


if __name__ == '__main__':
    main()