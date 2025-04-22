from matplotlib.pyplot import get
from data.data_utils import MergedDataset, Add_Old_Class_Mask
from data.fed_CXR import get_hete_federated_cxr_datasets
from data.fed_herbarium_19 import get_herbarium_datasets, get_hete_federated_herb_datasets
from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets, get_cifar_50_datasets, \
    get_cifar_90_10_datasets, get_cifar_80_20_datasets, get_cifar_70_30_datasets, get_cifar_60_40_datasets, \
    get_cifar_50_50_datasets, get_cifar_40_60_datasets, get_cifar_30_70_datasets, get_cifar_20_80_datasets, \
    get_cifar_10_90_datasets, get_federated_cifar_80_20_datasets
from data.fed_cifar import get_hete_federated_cifar_80_20_datasets, get_hete_federated_cifar100_datasets, get_hete_federated_cifar10_datasets
from data.fed_cub import get_hete_federated_cub_datasets, get_hete_federated_cub_datasets2
from data.stanford_cars import get_scars_datasets, get_hete_federated_scars_datasets
from data.imagenet import get_imagenet_100_datasets, get_hete_federated_imagenet_100_datasets
from data.cub import get_cub_datasets, get_federated_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.pets import get_pets_datasets, get_hete_federated_pets_datasets
from data.flower import get_flower_datasets
from data.food import get_food_datasets
from data.herbarium_19 import subsample_classes as subsample_dataset_herb
from data.cifar import subsample_classes as subsample_dataset_cifar
from data.stanford_cars import subsample_classes as subsample_dataset_scars
from data.imagenet import subsample_classes as subsample_dataset_imagenet
from data.cub import subsample_classes as subsample_dataset_cub
from data.fgvc_aircraft import subsample_classes as subsample_dataset_air
from data.pets import subsample_classes as subsample_dataset_pets
from data.flower import subsample_classes as subsample_dataset_flower
from data.food import subsample_classes as subsample_dataset_food
import numpy as np
from copy import deepcopy
import pickle
import os

from data.inaturalist import get_inaturalist_datasets

sub_sample_class_funcs = {
    'cifar10': subsample_dataset_cifar,
    'cifar100': subsample_dataset_cifar,
    'cifar50': subsample_dataset_cifar,
    'imagenet_100': subsample_dataset_imagenet,
    'herbarium_19': subsample_dataset_herb,
    'cub': subsample_dataset_cub,
    'aircraft': subsample_dataset_air,
    'scars': subsample_dataset_scars,
    'pets': subsample_dataset_pets,
    'flower': subsample_dataset_flower,
    'food': subsample_dataset_food
}

get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'hete_federated_cifar10': get_hete_federated_cifar10_datasets,
    'cifar100': get_cifar_100_datasets,
    'cifar50': get_cifar_50_datasets,
    'cifar90_10': get_cifar_90_10_datasets,
    'cifar80_20': get_cifar_80_20_datasets,
    'federated_cifar80_20': get_federated_cifar_80_20_datasets,
    'hete_federated_cifar80_20': get_hete_federated_cifar_80_20_datasets,
    'hete_federated_cifar100': get_hete_federated_cifar100_datasets,
    'cifar70_30': get_cifar_70_30_datasets,
    'cifar60_40': get_cifar_60_40_datasets,
    'cifar50_50': get_cifar_50_50_datasets,
    'cifar40_60': get_cifar_40_60_datasets,
    'cifar30_70': get_cifar_30_70_datasets,
    'cifar20_80': get_cifar_20_80_datasets,
    'cifar10_90': get_cifar_10_90_datasets,

    'imagenet_100': get_imagenet_100_datasets,
    'hete_federated_imagenet_100': get_hete_federated_imagenet_100_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'federated_cub': get_federated_cub_datasets,
    'hete_federated_cub': get_hete_federated_cub_datasets,
    'hete_federated_cxr': get_hete_federated_cxr_datasets,
    'hete2_federated_cub': get_hete_federated_cub_datasets2,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets,
    'hete_federated_scars': get_hete_federated_scars_datasets,
    'pets': get_pets_datasets,
    'hete_federated_pets': get_hete_federated_pets_datasets,
    'flower': get_flower_datasets,
    'food': get_food_datasets,
    'hete_federated_herb': get_hete_federated_herb_datasets,
}


def get_datasets(dataset_name, train_transform, test_transform, args):
    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform, federated_args=args,
                             train_classes=args.train_classes,
                             prop_train_labels=args.prop_train_labels,
                             split_train_val=False)

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    # target_transform = lambda x: target_transform_dict[x]
    label_rectify = False
    if sum(np.equal(np.sort(args.train_classes), np.unique(datasets['train_labelled'].targets))) < len(args.train_classes):
        label_rectify = True

    # ['train_labelled', 'train_unlabelled', 'val', 'test']
    for dataset_name, dataset in datasets.items():
        args.logger.info(f'dataset_name: {dataset_name}')
    for dataset_name, dataset in datasets.items():
        # if "train" in dataset_name or "val" in dataset_name or "test"in dataset_name:
        #     dataset.old_class_mask = [True  if _t in list(args.train_classes) else False for _t in datasets['train_labelled'].targets ]
        if dataset is not None and not isinstance(dataset, (dict, list, tuple, np.ndarray)):
            # [True if _t in list(args.train_classes) else False for _t in datasets['train_labelled'].targets ]
            if 'client' in dataset_name:
                dataset.target_transform = None
                # dataset.old_class_mask = [True if _t in list(dataset.client_train_labeled_classes) else False for _t in dataset.targets]

            # elif dataset_name == 'test':
            #     dataset.target_transform = None
            #     dataset.old_class_mask = [True if _t in list(dataset.client_train_labeled_classes) else False for _t in
            #                               dataset.targets]

            else:
                if label_rectify:
                    dataset.targets = [_t - 1 for _t in dataset.targets]

                dataset.old_class_mask = [True  if _t in list(args.train_classes) else False for _t in dataset.targets ]

            
            
            # if 'client' in dataset_name:
            #     dataset.target_transform = None
            # elif dataset_name == 'test':
            #     dataset.target_transform = None
            # elif 'train' in dataset_name:
            #     dataset.target_transform = None
            # else:
            #     dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    # train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
    #                               unlabelled_dataset=deepcopy(datasets['train_unlabelled']), re_label=False)
    assert sum(np.equal(np.sort(args.train_classes), np.unique(datasets['train_labelled'].targets))) == len(args.train_classes)
    
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']), re_label=True) # cub 200

    test_dataset = deepcopy(datasets['test'])
    test_dataset.transform = test_transform

    testset_unlabelled = deepcopy(datasets['test_unlabelled'])
    testset_unlabelled.transform = test_transform
    testset_labelled = deepcopy(datasets['test_labelled'])
    testset_labelled.transform = test_transform

    unlabelled_train_examples_test = Add_Old_Class_Mask(row_datset=deepcopy(datasets['train_unlabelled']),
                                                        labeled_or_not=0,
                                                        relabel_dict=train_dataset.relabel_dict,
                                                        transform=test_transform)
    # unlabelled_train_examples_test.transform = test_transform
    



    # unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    # unlabelled_train_examples_test.transform = test_transform

    
    unlabelled_train_examples_train = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_train.transform = train_transform

    labelled_train_examples = deepcopy(datasets['train_labelled'])
    labelled_train_examples.transform = test_transform

    labelled_train_examples_attribute = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                             unlabelled_dataset=deepcopy(datasets['train_unlabelled']), re_label=True)

    labelled_train_examples_attribute.transform = test_transform
    labelled_train_examples_attribute.labelled_dataset.transform = test_transform
    labelled_train_examples_attribute.unlabelled_dataset.transform = test_transform


    # if args.n_clients > 0:
    #     federated_train_datasets_dict = {}
    #     for i in range(args.n_clients):
    #         _federated_train = MergedDataset(labelled_dataset=deepcopy(datasets[f'federated_train_labelled_client-{i}']),
    #                                          unlabelled_dataset=deepcopy(datasets[f'federated_train_unlabelled_client-{i}']), re_label=True)

    #         # unlabelled_train_examples_test = Add_Old_Class_Mask(row_datset=deepcopy(datasets['train_unlabelled']),
    #         #                                                     labeled_or_not=0,
    #         #                                                     relabel_dict=train_dataset.relabel_dict,
    #         #                                                     transform=test_transform)

    #         _federated_train.transform = train_transform
    #         _federated_train.labelled_dataset.transform = train_transform
    #         _federated_train.unlabelled_dataset.transform = train_transform
    #         federated_train_datasets_dict[f'client-{i}-training'] = _federated_train
    #         _federated_clustering = MergedDataset(labelled_dataset=deepcopy(datasets[f'federated_train_labelled_client-{i}']),
    #                                          unlabelled_dataset=deepcopy(datasets[f'federated_train_unlabelled_client-{i}']), re_label=True)
    #         _federated_clustering.transform = test_transform
    #         _federated_clustering.labelled_dataset.transform = test_transform
    #         _federated_clustering.unlabelled_dataset.transform = test_transform
    #         federated_train_datasets_dict[f'client-{i}-clustering'] = _federated_clustering
    #         _federated_local_testing = deepcopy(datasets[f'federated_train_unlabelled_client-{i}'])
    #         # _federated_local_testing.transform = test_transform
    #         _federated_local_testing = Add_Old_Class_Mask(row_datset=_federated_local_testing,
    #                                                             labeled_or_not=0,
    #                                                             relabel_dict=_federated_train.relabel_dict,
    #                                                             transform=test_transform)
    #         federated_train_datasets_dict[f'client-{i}-local_testing'] = _federated_local_testing
    #          # = deepcopy(test_dataset)
    #         federated_train_datasets_dict[f'client-{i}-global_testing'] = Add_Old_Class_Mask(row_datset=deepcopy(test_dataset),
    #                                                       labeled_or_not=0,
    #                                                       relabel_dict=train_dataset.relabel_dict,
    #                                                       transform=test_transform)

    #     return train_dataset, test_dataset, unlabelled_train_examples_test, unlabelled_train_examples_train, datasets, labelled_train_examples, labelled_train_examples_attribute, federated_train_datasets_dict, testset_labelled, testset_unlabelled

    # else:
    #     return train_dataset, test_dataset, unlabelled_train_examples_test, unlabelled_train_examples_train, datasets, labelled_train_examples, labelled_train_examples_attribute
    return train_dataset, test_dataset, unlabelled_train_examples_test, unlabelled_train_examples_train, datasets, labelled_train_examples, labelled_train_examples_attribute

    #     return train_dataset, test_dataset, unlabelled_train_examples_test, unlabelled_train_examples_train, datasets, labelled_train_examples, labelled_train_examples_attribute, federated_train_datasets_dict, testset_labelled, testset_unlabelled
    #
    # else:
    #     return train_dataset, test_dataset, unlabelled_train_examples_test, unlabelled_train_examples_train, datasets, labelled_train_examples, labelled_train_examples_attribute


def get_class_splits(args):
    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    if args.dataset_name in ('federated_scars', 'federated_cub', 'federated_aircraft',
                             'hete_federated_scars', 'hete_federated_cub', 'hete2_federated_cub', 'hete_federated_aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10' or args.dataset_name == 'hete_federated_cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'cifar50':

        args.image_size = 32
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)



    elif args.dataset_name == 'cifar90_10':

        args.image_size = 32
        args.train_classes = range(90)
        args.unlabeled_classes = range(90, 100)

    elif args.dataset_name == 'cifar80_20':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'hete_federated_cifar80_20':
        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'hete_federated_cifar100':
        args.image_size = 32
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'federated_cifar80_20':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == 'cifar70_30':

        args.image_size = 32
        args.train_classes = range(70)
        args.unlabeled_classes = range(70, 100)

    elif args.dataset_name == 'cifar60_40':

        args.image_size = 32
        args.train_classes = range(60)
        args.unlabeled_classes = range(60, 100)

    elif args.dataset_name == 'cifar50_50':

        args.image_size = 32
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cifar40_60':

        args.image_size = 32
        args.train_classes = range(40)
        args.unlabeled_classes = range(40, 100)

    elif args.dataset_name == 'cifar30_70':

        args.image_size = 32
        args.train_classes = range(30)
        args.unlabeled_classes = range(30, 100)

    elif args.dataset_name == 'cifar20_80':

        args.image_size = 32
        args.train_classes = range(20)
        args.unlabeled_classes = range(20, 100)
    elif args.dataset_name == 'cifar10_90':

        args.image_size = 32
        args.train_classes = range(10)
        args.unlabeled_classes = range(10, 100)

    elif args.dataset_name == 'tinyimagenet':

        args.image_size = 64
        args.train_classes = range(100)
        args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == 'herbarium_19' or  args.dataset_name == 'hete_federated_herb':

        args.image_size = 224
        herb_path_splits = os.path.join(args.osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']


    elif args.dataset_name == 'imagenet_100' or args.dataset_name == 'hete_federated_imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'scars' or args.dataset_name == 'hete_federated_scars':

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

    elif args.dataset_name == 'cub' or args.dataset_name == 'federated_cub' or args.dataset_name == 'hete_federated_cub' or args.dataset_name == 'hete2_federated_cub':

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


    elif args.dataset_name == 'pets' or args.dataset_name == 'hete_federated_pets':

        args.image_size = 224
        args.train_classes = range(19)
        args.unlabeled_classes = range(19, 37)

    elif args.dataset_name == 'cxr' or args.dataset_name == 'hete_federated_cxr':

        args.image_size = 224
        args.train_classes = range(10)
        args.unlabeled_classes = range(10, 20)

    elif args.dataset_name == 'flower':

        args.image_size = 224
        args.train_classes = range(51)
        args.unlabeled_classes = range(51, 102)

    elif args.dataset_name == 'food':

        args.image_size = 224
        args.train_classes = range(51)
        args.unlabeled_classes = range(51, 101)

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

    else:

        raise NotImplementedError

    return args