from data.data_utils import MergedDataset

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.herbarium_19 import get_herbarium_datasets
from data.stanford_cars import get_scars_datasets
from data.imagenet import get_imagenet_100_datasets, get_imagenet_1k_datasets
from data.cub import get_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.inaturalist import get_inaturalist_datasets
from data.nabirds import get_nabirds_datasets

from copy import deepcopy
import pickle
import os

# from config import osr_split_dir


get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'cifar100_50': get_cifar_100_datasets,
    'cifar100_37': get_cifar_100_datasets,
    'cifar100_46': get_cifar_100_datasets,
    'cifar100_64': get_cifar_100_datasets,
    'cifar100_73': get_cifar_100_datasets,
    'cifar100_82': get_cifar_100_datasets,
    'imagenet_100': get_imagenet_100_datasets,
    'imagenet_1k': get_imagenet_1k_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets,
    #inaturalist
    'Actinopterygii': get_inaturalist_datasets,
    'Amphibia': get_inaturalist_datasets,
    'Animalia': get_inaturalist_datasets,
    'Arachnida': get_inaturalist_datasets,
    'Aves': get_inaturalist_datasets,
    'Chromista': get_inaturalist_datasets,
    'Fungi': get_inaturalist_datasets,
    'Insecta': get_inaturalist_datasets,
    'Mammalia': get_inaturalist_datasets,
    'Mollusca': get_inaturalist_datasets,
    'Plantae': get_inaturalist_datasets,
    'Protozoa': get_inaturalist_datasets,
    'Reptilia': get_inaturalist_datasets,
    #nabirds
    'nabirds': get_nabirds_datasets,
}


def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    valid_super_categories = ['Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Chromista', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia']

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]

    if dataset_name in valid_super_categories:
        datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform, subclassname=dataset_name,
                                train_classes=args.train_classes,
                                prop_train_labels=args.prop_train_labels,
                                split_train_val=False,
                                args=args)
    else:
        datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                                train_classes=args.train_classes,
                                prop_train_labels=args.prop_train_labels,
                                split_train_val=False,
                                args=args)

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    # test_dataset = datasets['test']
    # unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    # unlabelled_train_examples_test.transform = test_transform

    # return train_dataset, test_dataset, unlabelled_train_examples_test, datasets

    cluster_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                             unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    cluster_dataset.transform = test_transform
    cluster_dataset.labelled_dataset.transform = test_transform
    cluster_dataset.unlabelled_dataset.transform = test_transform
    return train_dataset, cluster_dataset


def get_class_splits(args):

    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'cifar100_50':

        args.image_size = 32
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cifar100_37':

        args.image_size = 32
        args.train_classes = range(70)
        args.unlabeled_classes = range(70, 100)

    elif args.dataset_name == 'cifar100_46':

        args.image_size = 32
        args.train_classes = range(60)
        args.unlabeled_classes = range(60, 100)

    elif args.dataset_name == 'cifar100_64':

        args.image_size = 32
        args.train_classes = range(40)
        args.unlabeled_classes = range(40, 100)

    elif args.dataset_name == 'cifar100_73':

        args.image_size = 32
        args.train_classes = range(30)
        args.unlabeled_classes = range(30, 100)

    elif args.dataset_name == 'cifar100_82':

        args.image_size = 32
        args.train_classes = range(20)
        args.unlabeled_classes = range(20, 100)

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(args.osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']

    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'imagenet_1k':

        args.image_size = 224
        args.train_classes = range(500)
        args.unlabeled_classes = range(500, 1000)
    
    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(args.osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(args.osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(args.osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == 'Actinopterygii':
        args.image_size = 224
        args.train_classes = range(27)
        args.unlabeled_classes = range(27, 53)

    elif args.dataset_name == 'Amphibia':
        args.image_size = 224
        args.train_classes = range(58)
        args.unlabeled_classes = range(58, 115)

    elif args.dataset_name == 'Animalia':
        args.image_size = 224
        args.train_classes = range(39)
        args.unlabeled_classes = range(39, 77)

    elif args.dataset_name == 'Arachnida':
        args.image_size = 224
        args.train_classes = range(28)
        args.unlabeled_classes = range(28, 56)

    elif args.dataset_name == 'Fungi':
        args.image_size = 224
        args.train_classes = range(61)
        args.unlabeled_classes = range(61, 121)

    elif args.dataset_name == 'Mammalia':
        args.image_size = 224
        args.train_classes = range(93)
        args.unlabeled_classes = range(93, 186)

    elif args.dataset_name == 'Mollusca':
        args.image_size = 224
        args.train_classes = range(47)
        args.unlabeled_classes = range(47, 93)

    elif args.dataset_name == 'Reptilia':
        args.image_size = 224
        args.train_classes = range(145)
        args.unlabeled_classes = range(145, 289)
    
    elif args.dataset_name == 'Plantae':
        args.image_size = 224
        args.train_classes = range(1051)
        args.unlabeled_classes = range(1051, 2101)    

    elif args.dataset_name == 'Insecta':
        args.image_size = 224
        args.train_classes = range(511)
        args.unlabeled_classes = range(511, 1021)  

    elif args.dataset_name == 'Aves':
        args.image_size = 224
        args.train_classes = range(482)
        args.unlabeled_classes = range(482, 964) 

    elif args.dataset_name == 'Chromista':
        args.image_size = 224
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 9)  

    elif args.dataset_name == 'Protozoa':
        args.image_size = 224
        args.train_classes = range(2)
        args.unlabeled_classes = range(2, 4)  

    elif args.dataset_name == 'nabirds':
        args.image_size = 224
        args.train_classes = range(278)
        args.unlabeled_classes = range(278, 555) 

    else:

        raise NotImplementedError

    return args
