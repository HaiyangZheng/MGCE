import os
import pandas as pd
import warnings
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, extract_archive

from data.data_utils import subsample_instances
import numpy as np
from copy import deepcopy

# class NABirds(VisionDataset):
#     """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.

#         Args:
#             root (string): Root directory of the dataset.
#             train (bool, optional): If True, creates dataset from training set, otherwise
#                creates from test set.
#             transform (callable, optional): A function/transform that  takes in an PIL image
#                and returns a transformed version. E.g, ``transforms.RandomCrop``
#             target_transform (callable, optional): A function/transform that takes in the
#                target and transforms it.
#             download (bool, optional): If true, downloads the dataset from the internet and
#                puts it in root directory. If dataset is already downloaded, it is not
#                downloaded again.
#     """
#     base_folder = 'nabirds/images'
#     filename = 'nabirds.tar.gz'
#     md5 = 'df21a9e4db349a14e2b08adfd45873bd'

#     def __init__(self, root, train=True, transform=None, target_transform=None, download=None):
#         super(NABirds, self).__init__(root, transform=transform, target_transform=target_transform)
#         if download is True:
#             msg = ("The dataset is no longer publicly accessible. You need to "
#                    "download the archives externally and place them in the root "
#                    "directory.")
#             raise RuntimeError(msg)
#         elif download is False:
#             msg = ("The use of the download flag is deprecated, since the dataset "
#                    "is no longer publicly accessible.")
#             warnings.warn(msg, RuntimeWarning)

#         dataset_path = os.path.join(root, 'nabirds')
#         if not os.path.isdir(dataset_path):
#             if not check_integrity(os.path.join(root, self.filename), self.md5):
#                 raise RuntimeError('Dataset not found or corrupted.')
#             extract_archive(os.path.join(root, self.filename))
#         self.loader = default_loader
#         self.train = train

#         image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
#                                   sep=' ', names=['img_id', 'filepath'])
#         image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
#                                          sep=' ', names=['img_id', 'target'])
#         # Since the raw labels are non-continuous, map them to new ones
#         self.label_map = get_continuous_class_map(image_class_labels['target'])
#         train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
#                                        sep=' ', names=['img_id', 'is_training_img'])
#         data = image_paths.merge(image_class_labels, on='img_id')
#         self.data = data.merge(train_test_split, on='img_id')
#         # Load in the train / test split
#         if self.train:
#             self.data = self.data[self.data.is_training_img == 1]
#         else:
#             self.data = self.data[self.data.is_training_img == 0]

#         # Load in the class data
#         self.class_names = load_class_names(dataset_path)
#         self.class_hierarchy = load_hierarchy(dataset_path)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data.iloc[idx]
#         path = os.path.join(self.root, self.base_folder, sample.filepath)
#         target = self.label_map[sample.target]
#         img = self.loader(path)

#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return img, target

class NABirds(VisionDataset):
    """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'nabirds/images'
    filename = 'nabirds.tar.gz'
    md5 = 'df21a9e4db349a14e2b08adfd45873bd'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=None):
        super(NABirds, self).__init__(root, transform=transform, target_transform=target_transform)
        if download is True:
            msg = ("The dataset is no longer publicly accessible. You need to "
                   "download the archives externally and place them in the root "
                   "directory.")
            raise RuntimeError(msg)
        elif download is False:
            msg = ("The use of the download flag is deprecated, since the dataset "
                   "is no longer publicly accessible.")
            warnings.warn(msg, RuntimeWarning)

        dataset_path = os.path.join(root, 'nabirds')
        if not os.path.isdir(dataset_path):
            if not check_integrity(os.path.join(root, self.filename), self.md5):
                raise RuntimeError('Dataset not found or corrupted.')
            extract_archive(os.path.join(root, self.filename))
        self.loader = default_loader
        self.train = train

        image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        # Since the raw labels are non-continuous, map them to new ones
        self.label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        
        # Convert relative paths to complete paths
        image_paths['filepath'] = image_paths['filepath'].apply(
            lambda x: os.path.join(self.root, self.base_folder, x)
        )
        
        data = image_paths.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')
        
        
        # Load in the train / test split
        if self.train:
            self.samples = data[data.is_training_img == 1]
        else:
            self.samples = data[data.is_training_img == 0]

        # Load in the class data
        self.class_names = load_class_names(dataset_path)
        self.class_hierarchy = load_hierarchy(dataset_path)

        self.data = self.samples['filepath'].tolist()
        self.targets = [self.label_map[target] for target in self.samples['target'].values]
        self.uq_idxs = np.array(range(len(self.samples)))  # Unique indices for each sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        path = sample.filepath  # filepath is now complete path
        target = self.label_map[sample.target]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, self.uq_idxs[idx]


def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}


def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names


def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents

##----------------------------------------------------GCD setting-----------------------------------------------##
def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = dataset.samples[mask]
    
    dataset.data = np.array(dataset.data)[mask]
    dataset.targets = np.array(dataset.targets)[mask].tolist()  # 改为使用mask
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset

def subsample_classes(dataset, include_classes=range(160)):

    cls_idxs = [idx for idx, class_index in enumerate(dataset.targets) if class_index in include_classes]

    # TODO: For now have no target transform
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.targets)

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

def get_nabirds_datasets(train_transform, test_transform, train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0, args=None):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = NABirds(root=args.nabirds_root, transform=train_transform, train=True)

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
    test_dataset = NABirds(root=args.nabirds_root, transform=test_transform, train=False)

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




# if __name__ == '__main__':
#     nabirds_root='/leonardo_work/IscrC_DiffGRE/Datasets/NABirds'
#     train_dataset = NABirds(root=nabirds_root, train=True, download=False)
#     test_dataset = NABirds(root=nabirds_root, train=False, download=False)

#     print(set(train_dataset.targets))

#     print("===== Dataset Info =====")
#     print(f"Train size: {len(train_dataset)} samples")
#     print(f"Test size:  {len(test_dataset)} samples")

#     # 打印类别数
#     num_classes = len(train_dataset.label_map)
#     print(f"Number of classes: {num_classes}")

if __name__ == '__main__':
    args = type('', (), {})()  # Create a dummy args object
    args.nabirds_root = '/leonardo_work/IscrC_DiffGRE/Datasets/NABirds'

    x = get_nabirds_datasets(None, None, split_train_val=False,
                         train_classes=range(278), prop_train_labels=0.5, args=args)

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

    # print(x['train_labelled'].targets)
    # print(x['train_labelled'].data)
    for i in range(10):
        print(x['train_labelled'].data[i], x['train_labelled'].targets[i], x['train_labelled'].uq_idxs[i])