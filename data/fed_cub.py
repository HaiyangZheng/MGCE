import os
import pandas as pd
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from data.data_utils import subsample_instances, dirichlet_split_noniid
# from config import cub_root
import torch
import torch.nn.functional as F
class CustomCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=True):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train


        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')

        name_attribute = pd.read_csv(os.path.join(self.root, 'attributes.txt'), sep=' ',
                                     names=['attribute_id', 'attribute_name'])
        from collections import defaultdict
        dict_attribute = defaultdict(list)
        for _i, _name in zip(name_attribute['attribute_id'], name_attribute['attribute_name']):
            dict_attribute[_name.split('::')[0]].append(_name.split('::')[1])

        names_ = list(dict_attribute.keys())
        processed_attribute_file = os.path.join(self.root,'CUB_200_2011','processed_attributes.txt')
        A_all = pd.read_csv(processed_attribute_file, sep=' ', names=names_)
        A_all.insert(0, 'img_id', list(range(1, len(A_all)+1)))
        self.data = data.merge(A_all, on='img_id')
        self.dict_attribute = dict_attribute
        class_attributes_file = os.path.join(self.root,'CUB_200_2011','attributes',
                                             'class_attribute_labels_continuous.txt')
        C_A = np.zeros((200, 312))
        class_attr_rf = open(class_attributes_file, 'r')
        i = 0
        for line in class_attr_rf.readlines():
            strs = line.strip().split(' ')
            for j in range(len(strs)):
                C_A[i][j] = 0.0 if strs[j] == '0.0' else float(strs[j]) * 0.01
            i += 1
        class_attr_rf.close()
        # C_A_tensor = torch.from_numpy(C_A)
        # C_A_tensor = F.normalize(C_A_tensor)
        # diag = C_A_tensor.matmul(C_A_tensor.t())
        # show = diag.numpy()
        # import seaborn as sns
        # matric = sns.heatmap(show, annot=False)
        # figure = matric.get_figure()
        # figure.savefig('class_confuse_matric.png', dpi=400)
        if self.train:
            self.data = self.data[self.data.is_training_img == 1].to_numpy()
        else:
            self.data = self.data[self.data.is_training_img == 0].to_numpy()
        self.targets = [-1] * len(self.data)
        for index, row in enumerate(self.data):
            self.data[index][1] = os.path.join(self.root, self.base_folder, row[1])
            self.targets[index] = row[2]
        # print(self.data)

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for row in self.data:
            filepath = row[1]
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = self.data.iloc[idx]
        sample = self.data[idx]
        path = sample[1]
        # target = int(sample[2]) - 1  # Targets start at 1 by default, so shift to 0
        target = self.targets[idx]
        img = self.loader(path)
        # attribute = sample[4:].to_numpy(dtype=np.int32)
        attribute = np.array(sample[4:].tolist())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[idx], attribute


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]
    temp = deepcopy(dataset.targets)
    temp_target = np.array(temp).astype(np.int64)
    dataset.targets = temp_target[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    # cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]
    cls_idxs = []
    target_idxs = []
    for x, r in enumerate(dataset.data):
        if int(r[2]) in include_classes_cub:
            cls_idxs.append(x)
    #         target_idxs.append(int(r[2]))
    # target_idxs, cls_idxs = np.array(target_idxs), np.array(cls_idxs)
    # delete_index = []
    # print(len(cls_idxs))
    # for target in np.unique(target_idxs):
    #     if len(np.where(target_idxs == target)[0]) == 1:
    #         delete_index.extend(np.where(target_idxs == target)[0])
    # np.delete(cls_idxs, delete_index, None)
    # print(len(cls_idxs))
    # TODO: For now have no target transform because deepcopy cannot copy lambda
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i


    dataset = subsample_dataset(dataset, cls_idxs)
    if len(target_xform_dict) > 0:
        dataset.target_xform_dict = target_xform_dict
        dataset.target_transform = lambda x: target_xform_dict[x]
    else:
        dataset.target_xform_dict = None
        dataset.target_transform = None

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    # train_classes = np.unique(train_dataset.data['target'])
    train_classes = np.unique(train_dataset.data[:, 2])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        # cls_idxs = np.where(train_dataset.data['target'] == cls)[0]
        cls_idxs = np.where(train_dataset.data[:, 2] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs



# back up for hetero labeled diata 0.001
def get_hete_federated_cub_datasets(train_transform, test_transform, federated_args,  train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCub2011(root=federated_args.cub_root, transform=train_transform, train=True, download=False)
    train_labels = np.array(deepcopy(whole_training_set.targets)) - 1
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
        client_train_dataset.targets = np.array(client_train_dataset.targets) - 1 # align GCD
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
        test_dataset = CustomCub2011(root=federated_args.cub_root, transform=test_transform, train=False, download=False)
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
    whole_test_dataset = CustomCub2011(root=federated_args.cub_root, transform=test_transform, train=False,
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















# first label and unlabel then \beta unlabel can not cover all labeled class
def get_hete_federated_cub_datasets2(train_transform, test_transform, federated_args,  train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCub2011(root=federated_args.cub_root, transform=train_transform, train=True, download=False)
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
                                         n_clients=federated_args.n_clients)

    client_idcs_train_unlabeled = dirichlet_split_noniid(train_labels=train_dataset_unlabelled.targets,
                                                       alpha=federated_args.dirichlet,
                                                       n_clients=federated_args.n_clients)

    ##### visulization statistic ##############
    # Get labelled training set which has subsampled classes, then subsample some indices from that

    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # # for label_idcs, unlabel_idcs in zip(client_idcs_train_labeled, client_idcs_train_unlabeled):
    # train_labels_set = set(train_dataset_labelled.targets)
    # plt.figure(figsize=(20, 3))
    # plt.hist([train_dataset_labelled.targets[idc] for idc in client_idcs_train_labeled], stacked=True,
    #          bins=np.arange(min(train_dataset_labelled.targets) - 0.5, max(train_dataset_labelled.targets) + 1.5, 1),
    #          label=["Client {}".format(i) for i in range(federated_args.num_clients)], rwidth=0.5)
    # plt.xticks(np.arange(len(train_labels_set)), train_labels_set)
    # plt.legend()
    # plt.show()
    #
    # train_labels_set = set(train_dataset_unlabelled.targets)
    # plt.figure(figsize=(20, 3))
    # plt.hist([train_dataset_unlabelled.targets[idc] for idc in client_idcs_train_unlabeled], stacked=True,
    #          bins=np.arange(min(train_dataset_unlabelled.targets) - 0.5, max(train_dataset_unlabelled.targets) + 1.5, 1),
    #          label=["Client {}".format(i) for i in range(federated_args.num_clients)], rwidth=0.5)
    # plt.xticks(np.arange(len(train_labels_set)), train_labels_set)
    # plt.legend()
    # plt.show()

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
        print(f'The number of labeled and unlabeled classes in the {_i}-th client: {len(set(__set)|set(unlabeled_set_list[_i]))}')
        info_dict.update({f'client-{_i}': (len(set(__set)),
                                           len(set(unlabeled_set_list[_i])),
                                           len(labeled_set_list[_i]),
                                           len(unlabeled_set_list[_i]),
                                           len(set(__set) | set(unlabeled_set_list[_i])))})
        labeled_unit_set = labeled_unit_set | set(__set)
        unlabeled_unit_set = unlabeled_unit_set | set(unlabeled_set_list[_i])

    print(f'The labeled {len(labeled_unit_set)} classes are totally include across all clients: {labeled_unit_set}')
    info_dict.update({'global seen classes': labeled_unit_set})
    info_dict.update({'num global classes': 200})
    print(f'The unlabeled {len(unlabeled_unit_set)} classes are totally include across all clients: {unlabeled_unit_set}')
    count_dict = {}
    for i in list(labeled_unit_set):
        count_dict[i] = len([i for _list in labeled_set_list if i in _list])
    n = 0
    for v in count_dict.values():
        if v >= 2:
            n = n + 1
    print(f'shared across at least 2, <{n}> classes')
    info_dict.update({'total labeled classes': labeled_unit_set})
    info_dict.update({'total unlabeled classes': unlabeled_unit_set})
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


    for index_client in range(federated_args.n_clients):
        train_dataset_labelled = subsample_dataset(deepcopy(whole_training_set), client_idcs_train_labeled[index_client])

        # subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
        # train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
        # Split into training and validation sets
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

        # Get unlabelled data

        train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), client_idcs_train_unlabeled[index_client])
        # Get test set for all classes
        test_dataset = CustomCub2011(root=federated_args.cub_root, transform=test_transform, train=False, download=False)
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
    whole_test_dataset = CustomCub2011(root=federated_args.cub_root, transform=test_transform, train=False, download=False)

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



# back up for caculate
def get_hete_federated_cub_datasets3(train_transform, test_transform, federated_args,  train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = CustomCub2011(root=federated_args.cub_root, transform=train_transform, train=True, download=False)
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
        print(f'The number of classes in the {_i}-th client: {len(__set)}')
        print(f'The classes in the {_i}-th client: {__set}')
        print(f'The number of images in the {_i}-th client: {len(client_idcs[_i])}')
        info_dict.update({f'client-{_i}': (len(__set),
                                           __set,
                                           len(client_idcs[_i]),
                                           client_idcs[_i])})
        unit_set = unit_set | __set
    print(f'The classes are totally include in across all clients: {unit_set}')

    info_dict.update({'total classes': unit_set})
    for _set in set_list:
        unit_set = _set & unit_set
    print(f'The classes shared across all clients: {unit_set}')

    info_dict.update({'shared classes': unit_set})
    federated_test_dataset = {}
    federated_train_dataset_labelled = {}
    federated_train_dataset_unlabelled = {}
    federated_train_dataset_val = {}
    federated_dataset_information = {}

    for index_client in range(federated_args.n_clients):
        client_train_dataset = subsample_dataset(deepcopy(whole_training_set), client_idcs[index_client])
        client_train_labeled_classes = np.random.choice(np.unique(client_train_dataset.targets), len(set(client_train_dataset.targets))//2)
        client_train_labeled_classes = client_train_labeled_classes - 1 # align GCD
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
        test_dataset = CustomCub2011(root=federated_args.cub_root, transform=test_transform, train=False, download=False)
        train_dataset_labelled_split = train_dataset_labelled_split if split_train_val else train_dataset_labelled
        federated_train_dataset_labelled[f'client-{index_client}'] = deepcopy(train_dataset_labelled_split)
        val_dataset_labelled_split = val_dataset_labelled_split if split_train_val else None
        federated_train_dataset_val[f'client-{index_client}'] = deepcopy(val_dataset_labelled_split)
        federated_test_dataset[f'client-{index_client}'] = deepcopy(test_dataset)
        federated_train_dataset_unlabelled[f'client-{index_client}'] = deepcopy(train_dataset_unlabelled)

    labeled_class_set_each_client_list = [federated_train_dataset_labelled[f'client-{index_client}'].targets.tolist() for index_client in range(federated_args.n_clients)]

    unit_set = set()
    for _i, __set in enumerate(labeled_class_set_each_client_list):
        print(f'The number of classes in labeled data of the {_i}-th client: {len(set(__set))}')
        print(f'The classes in labeled data of the {_i}-th client: {set(__set)}')
        print(f'The number of labeled images in the {_i}-th client: {__set}')
        info_dict.update({f'labelled_client-{_i}': (len(set(__set)),
                                           set(__set),
                                           len(__set),
                                           __set)})
        unit_set = unit_set | set(__set)
    print(f'The labeled classes are totally include in across all clients: {unit_set}')

    info_dict.update({'total labeled classes': unit_set})
    for _set in set_list:
        unit_set = _set & unit_set
    print(f'The labeled classes shared across all clients: {unit_set}')

    info_dict.update({'labeled shared classes': unit_set})

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
    test_dataset = CustomCub2011(root=federated_args.cub_root, transform=test_transform, train=False, download=False)
    whole_test_dataset = CustomCub2011(root=federated_args.cub_root, transform=test_transform, train=False,
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
    print(test_info_update_dict)
    info_dict.update(test_info_update_dict)
    test_dataset.information = info_dict


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

    x = get_hete_federated_cub_datasets(None, None, split_train_val=False,
                         train_classes=range(100), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].data["target"].values))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].data["target"].values))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')

if __name__ == '__main__':
    main()