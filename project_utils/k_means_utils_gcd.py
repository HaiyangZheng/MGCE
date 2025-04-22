import argparse
from asyncio.log import logger
import os

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from project_utils.cluster_evaluate_utils import log_accs_from_preds
from project_utils.finch_utils import FINCH, semi_FINCH
from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans

from tqdm import tqdm

# TODO: Debug
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch.nn as nn


def test_kmeans_semi_sup(model, test_loader, epoch, save_name, device, args, K=None, logger_class=None, in_training=False):
    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """

    if K is None:
        K = args.num_labeled_classes + args.num_unlabeled_classes

    logger_class(f'set k = {K} in ssk')
    if isinstance(model, (list, tuple, nn.ModuleList, nn.ModuleDict)) and len(model) >= 2:
        co_feat_extractor, att_feat_extractor = model[0], model[1]
        co_feat_extractor.eval()
        att_feat_extractor.eval()

        all_feats, all_co_feats = [], []
        targets = np.array([])
        mask_lab = np.array([])  # From all the data, which instances belong to the labelled set
        mask_cls = np.array([])  # From all the data, which instances belong to Old classes

        print('Collating features...')
        # First extract all features
        for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(tqdm(test_loader)):
            images = images.to(device)

            co_feats, att_embs = co_feat_extractor(images)
            att_feats = att_feat_extractor(att_embs)
            if args.use_l2_in_ssk:
                co_feats = torch.nn.functional.normalize(co_feats, dim=-1)
                att_feats = torch.nn.functional.normalize(att_feats, dim=-1)

            feats = torch.cat((co_feats, att_feats), dim=1)
            all_co_feats.append(co_feats.cpu().numpy())
            all_feats.append(feats.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                     else False for x in label]))
            mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

        # -----------------------
        # K-MEANS
        # -----------------------
        mask_lab = mask_lab.astype(bool)
        mask_cls = mask_cls.astype(bool)

        all_feats = np.concatenate(all_feats)
        all_co_feats = np.concatenate(all_co_feats)

        l_feats = all_feats[mask_lab]  # Get labelled set
        u_feats = all_feats[~mask_lab]  # Get unlabelled set
        l_co_feats = all_co_feats[mask_lab]  # Get labelled set
        u_co_feats = all_co_feats[~mask_lab]  # Get unlabelled set
        l_targets = targets[mask_lab]  # Get labelled targets
        u_targets = targets[~mask_lab]  # Get unlabelled targets

        print('Fitting Semi-Supervised K-Means with concatenated_feature...')
        if in_training:
            max_kmeans_iter = args.train_max_kmeans_iter
        else:
            max_kmeans_iter = args.max_kmeans_iter
        logger_class('max_kmeans_iter: {max_kmeans_iter}!')
        kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=max_kmeans_iter, init='k-means++',
                               n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=512,
                               mode=None)

        l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                                  x in (l_feats, u_feats, l_targets, u_targets))

        kmeans.fit_mix(u_feats, l_feats, l_targets)
        all_preds = kmeans.labels_.cpu().numpy()
        u_targets = u_targets.cpu().numpy()

        # -----------------------
        # EVALUATE
        # -----------------------
        # Get preds corresponding to unlabelled set
        preds = all_preds[~mask_lab]

        # Get portion of mask_cls which corresponds to the unlabelled set
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)

        # -----------------------
        # EVALUATE
        # -----------------------
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask,
                                                        eval_funcs=args.eval_funcs,
                                                        save_name=save_name, T=epoch, print_output=True)
        logger_class(
            'Using concatenated_feature ==> SS-K Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc,
                                                                                                          old_acc,
                                                                                                          new_acc))

        print('Using contrastive_feature Fitting Semi-Supervised K-Means...')
        kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                               n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=512,
                               mode=None)

        l_c0_feats, u_co_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                                        x in (l_co_feats, u_co_feats, l_targets, u_targets))

        kmeans.fit_mix(u_co_feats, l_c0_feats, l_targets)
        all_preds = kmeans.labels_.cpu().numpy()
        u_targets = u_targets.cpu().numpy()

        # -----------------------
        # EVALUATE
        # -----------------------
        # Get preds corresponding to unlabelled set
        preds = all_preds[~mask_lab]

        # Get portion of mask_cls which corresponds to the unlabelled set
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)

        # -----------------------
        # EVALUATE
        # -----------------------
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask,
                                                        eval_funcs=args.eval_funcs,
                                                        save_name=save_name, T=epoch, print_output=True)

        return all_acc, old_acc, new_acc, kmeans


    else:
        model.eval()
        all_feats = []
        targets = np.array([])
        mask_lab = np.array([])  # From all the data, which instances belong to the labelled set
        mask_cls = np.array([])  # From all the data, which instances belong to Old classes

        print('Collating features...')
        # First extract all features
        for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            feats = model(images)
            if args.use_l2_in_ssk:
                feats = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.detach().cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                     else False for x in label]))
            mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

        # -----------------------
        # K-MEANS
        # -----------------------
        mask_lab = mask_lab.astype(bool)
        mask_cls = mask_cls.astype(bool)

        all_feats = np.concatenate(all_feats)

        l_feats = all_feats[mask_lab]  # Get labelled set
        u_feats = all_feats[~mask_lab]  # Get unlabelled set
        l_targets = targets[mask_lab]  # Get labelled targets
        u_targets = targets[~mask_lab]  # Get unlabelled targets
        if in_training:
            max_kmeans_iter = args.train_max_kmeans_iter
        else:
            max_kmeans_iter = args.max_kmeans_iter
        logger_class(f'Fitting Semi-Supervised K-Means... max_kmeans_iter = {max_kmeans_iter}')
        kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                               n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=512,
                               mode=None)

        l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                                  x in (l_feats, u_feats, l_targets, u_targets))

        kmeans.fit_mix(u_feats, l_feats, l_targets)
        all_preds = kmeans.labels_.cpu().numpy()
        u_targets = u_targets.cpu().numpy()

        # -----------------------
        # EVALUATE
        # -----------------------
        # Get preds corresponding to unlabelled set
        preds = all_preds[~mask_lab]

        # Get portion of mask_cls which corresponds to the unlabelled set
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)

        # -----------------------
        # EVALUATE
        # -----------------------
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask,
                                                        eval_funcs=args.eval_funcs,
                                                        save_name=save_name, T=epoch, print_output=True)

        if args.use_contrastive_cluster and args.contrastive_cluster_method == 'ssk':
            return all_acc, old_acc, new_acc, kmeans, all_feats
        else:
            return all_acc, old_acc, new_acc, kmeans, None


def test_semi_finch(model, test_loader,
                epoch, save_name, device,
                args, logger_class=None, output_kmeans=False, index_client=None, hyper_K=None):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating features...')

    # First extract all features
    for batch_idx, _item in enumerate(tqdm(test_loader)):
        images = _item[0]
        label = _item[1]
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        # concat_feats = model(images, concat=True)
        if index_client is not None:
            feats = model.feature_extract(images)
            feats = model.client_model_dict[f'client-{index_client}'](feats)
        else:
            feats = model(images, concat=False)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())

        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    all_feats = np.concatenate(all_feats)

    if hyper_K is not None:
        hyperparameter_K = hyper_K
    else:
        hyperparameter_K = args.num_labeled_classes + args.num_unlabeled_classes
    logger_class(f'Using sime_FINCH with {args.semi_finch_mode} mode to Clustering... k={hyperparameter_K}')
    # for _mode in ['hard', 'random', 'easy', 'none']:
    #     c, num_clust, req_c = semi_FINCH(all_feats, targets, mask, mode=_mode,
    #                                      req_clust=hyperparameter_K, distance='cosine',verbose=True)
    #     preds = req_c
    #     all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
    #                                                     T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
    #                                                     writer=args.writer)
    #     logger_class(f'semi_FINCH with {_mode} mode: ALL {all_acc}\t OLD {old_acc}\t NEW {new_acc}')

    c, num_clust, req_c = semi_FINCH(all_feats, targets, mask, mode=args.semi_finch_mode, req_clust=hyperparameter_K, distance='cosine',
                                verbose=True)
    preds = req_c


    logger_class('Using semi_FINCH Done')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)



    if output_kmeans:
        return all_acc, old_acc, new_acc, req_c, all_feats
    else:
        return all_acc, old_acc, new_acc

def test_kmeans_and_semi_finch(model, test_loader,
                epoch, save_name, device,
                args, logger_class=None, output_kmeans=False, index_client=None, hyper_K=None):
    model.eval()

    all_feats = []
    all_concat_feats = []
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating features...')

    # First extract all features
    for batch_idx, _item in enumerate(tqdm(test_loader)):
        images = _item[0]
        label = _item[1]
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        # concat_feats = model(images, concat=True)
        if index_client is not None:
            feats = model.feature_extract(images)
            feats = model.client_model_dict[f'client-{index_client}'](feats)
        else:
            feats = model(images, concat=False)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # all_feats.append(feats.cpu().detach().numpy())
        # targets = np.append(targets, label.cpu().detach().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    all_feats = np.concatenate(all_feats)

    if hyper_K is not None:
        hyperparameter_K = hyper_K
    else:
        hyperparameter_K = args.num_labeled_classes + args.num_unlabeled_classes
    logger_class(f'Using sime_FINCH with {args.semi_finch_mode} mode to Clustering... k={hyperparameter_K}')
    c, num_clust, req_c = semi_FINCH(all_feats, targets, mask, mode=args.semi_finch_mode, req_clust=hyperparameter_K,
                                     distance='cosine',
                                     verbose=True)
    preds = req_c

    optimal_hierarchical = np.argmin(np.abs(np.array(num_clust) - hyperparameter_K))

    logger_class(
        f"use simi-finch to estimate cluster number: {num_clust}, selected: {num_clust[optimal_hierarchical]}, truth: {hyperparameter_K}")

    preds_finch = c[:, optimal_hierarchical]



    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)
    all_acc_finch_estimated, old_acc_finch_estimated, new_acc_finch_estimated = log_accs_from_preds(y_true=targets, y_pred=preds_finch, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)
    logger_class(
        'Semi-Finch testing result for Non-overlapping whole test set K-means Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(
            all_acc_finch_estimated, old_acc_finch_estimated, new_acc_finch_estimated))
    logger_class('Using semi_FINCH Done')
    logger_class(f'Using contrastive feature Fitting K-Means... k={hyperparameter_K}')
    kmeans = KMeans(n_clusters=hyperparameter_K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    logger_class('Using contrastive feature Done')

    # -----------------------
    # EVALUATE
    # -----------------------
    k_all_acc, k_old_acc, k_new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    if output_kmeans:
        return all_acc, old_acc, new_acc, kmeans, all_feats
    else:
        return all_acc, old_acc, new_acc, k_all_acc, k_old_acc, k_new_acc


def test_kmeans_and_finch(model, test_loader,
                epoch, save_name, device,
                args, logger_class=None, output_kmeans=False, index_client=None, hyper_K=None):
    model.eval()

    all_feats = []
    all_concat_feats = []
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating features...')

    # First extract all features
    for batch_idx, _item in enumerate(tqdm(test_loader)):
        images = _item[0]
        label = _item[1]
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        # concat_feats = model(images, concat=True)
        if index_client is not None:
            feats = model.feature_extract(images)
            feats = model.client_model_dict[f'client-{index_client}'](feats)
        else:
            feats = model(images, concat=False)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # all_feats.append(feats.cpu().detach().numpy())
        # targets = np.append(targets, label.cpu().detach().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # logger_class('Using no_l2 feature Fitting K-Means...')
    #
    # all_concat_feats = np.concatenate(all_concat_feats)
    #
    # kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(
    #     all_concat_feats)
    # preds = kmeans.labels_
    #
    # logger_class('Using no_l2 feature Done')
    #
    #
    # # -----------------------
    # # EVALUATE
    # # -----------------------
    # all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
    #                                                 T=epoch, eval_funcs=args.eval_funcs, save_name=save_name + 'add',
    #                                                 writer=args.writer)
    # logger_class('no_l2 feature kmeans Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
    #                                                                                     new_acc))
    all_feats = np.concatenate(all_feats)

    if hyper_K is not None:
        hyperparameter_K = hyper_K
    else:
        hyperparameter_K = args.num_labeled_classes + args.num_unlabeled_classes
    logger_class(f'Using FINCH to Clustering... k={hyperparameter_K}')
    c, num_clust, req_c = FINCH(all_feats, initial_rank=None, req_clust=hyperparameter_K, distance='cosine',
                                verbose=True)
    preds = req_c

    logger_class('Using FINCH Done')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    logger_class(f'Using contrastive feature Fitting K-Means... k={hyperparameter_K}')
    kmeans = KMeans(n_clusters=hyperparameter_K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    logger_class('Using contrastive feature Done')

    # -----------------------
    # EVALUATE
    # -----------------------
    k_all_acc, k_old_acc, k_new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    if output_kmeans:
        return all_acc, old_acc, new_acc, kmeans, all_feats
    else:
        return all_acc, old_acc, new_acc, k_all_acc, k_old_acc, k_new_acc


def test_kmeans(model, test_loader,
                epoch, save_name, device,
                args, logger_class=None, output_kmeans=False, index_client=None, hyper_K=None):
    model.eval()

    all_feats = []
    all_concat_feats = []
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating features...')

    # First extract all features
    for batch_idx, _item in enumerate(tqdm(test_loader)):
        images = _item[0]
        label = _item[1]
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        # concat_feats = model(images, concat=True)
        if index_client is not None:
            feats = model.feature_extract(images)
            feats = model.client_model_dict[f'client-{index_client}'](feats)
        else:
            feats = model(images, concat=False)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # all_feats.append(feats.cpu().detach().numpy())
        # targets = np.append(targets, label.cpu().detach().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # logger_class('Using no_l2 feature Fitting K-Means...')
    #
    # all_concat_feats = np.concatenate(all_concat_feats)
    #
    # kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(
    #     all_concat_feats)
    # preds = kmeans.labels_
    #
    # logger_class('Using no_l2 feature Done')
    #
    #
    # # -----------------------
    # # EVALUATE
    # # -----------------------
    # all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
    #                                                 T=epoch, eval_funcs=args.eval_funcs, save_name=save_name + 'add',
    #                                                 writer=args.writer)
    # logger_class('no_l2 feature kmeans Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
    #                                                                                     new_acc))
    all_feats = np.concatenate(all_feats)

    if hyper_K is not None:
        hyperparameter_K = hyper_K
    else:
        hyperparameter_K = args.num_labeled_classes + args.num_unlabeled_classes
    logger_class(f'Using contrastive feature Fitting K-Means... k={hyperparameter_K}')
    kmeans = KMeans(n_clusters=hyperparameter_K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    logger_class('Using contrastive feature Done')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    if output_kmeans:
        return all_acc, old_acc, new_acc, kmeans, all_feats
    else:
        return all_acc, old_acc, new_acc


def test_kmeans_k(model, test_loader,
                epoch, save_name, device,
                args, K=None, logger_class=None, output_kmeans=False):
    model.eval()
    if K is None:
        K = args.num_labeled_classes + args.num_unlabeled_classes
    logger_class(f'kmean k = {K}')
    all_feats = []
    all_concat_feats = []
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating features...')

    # First extract all features
    for batch_idx, _item in enumerate(tqdm(test_loader)):
        images = _item[0]
        label = _item[1]
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        # concat_feats = model(images, concat=True)
        feats = model(images, concat=False)
        concat_feats = feats.detach().clone()
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_concat_feats.append(concat_feats.cpu().numpy())
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # all_feats.append(feats.cpu().detach().numpy())
        # targets = np.append(targets, label.cpu().detach().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    logger_class('Using no_l2 feature Fitting K-Means...')

    all_concat_feats = np.concatenate(all_concat_feats)

    kmeans = KMeans(n_clusters=K, random_state=0).fit(
        all_concat_feats)
    preds = kmeans.labels_

    logger_class('Using no_l2 feature Done')


    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name + 'add',
                                                    writer=args.writer)
    logger_class('no_l2 feature kmeans Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))

    logger_class('Using contrastive feature Fitting K-Means...')


    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    logger_class('Using contrastive feature Done')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    if output_kmeans:
        return all_acc, old_acc, new_acc, kmeans, all_feats
    else:
        return all_acc, old_acc, new_acc


def test_memory_buffer(test_loader, predicted_label,
                save_name, epoch,
                args, logger_class=None):

    preds = predicted_label
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating real label...')

    # First extract all features
    for batch_idx, _item in enumerate(tqdm(test_loader)):
        label = _item[1]
        if isinstance(label, int):
            targets = np.append(targets, np.array([label]))
            mask = np.append(mask, np.array([True if label in range(len(args.train_classes)) else False]))
        else:
            targets = np.append(targets, label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                             else False for x in label]))
    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name + 'add',
                                                    writer=args.writer)
    logger_class('Direct test from Memory Buffer Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))


    return all_acc, old_acc, new_acc

def test_kmeans_save(model, test_loader,
                epoch, save_name, device,
                args, logger_class=None, output_kmeans=False):
    model.eval()

    all_feats = []
    all_concat_feats = []
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating features...')

    # First extract all features
    for batch_idx, _item in enumerate(tqdm(test_loader)):
        images = _item[0]
        label = _item[1]
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        # concat_feats = model(images, concat=True)
        feats = model(images, concat=False)
        concat_feats = feats.detach().clone()
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_concat_feats.append(concat_feats.cpu().numpy())
        all_feats.append(feats.detach().cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # all_feats.append(feats.cpu().detach().numpy())
        # targets = np.append(targets, label.cpu().detach().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    logger_class('Using no_l2 feature Fitting K-Means...')

    all_feats = np.concatenate(all_feats)
    feature_save_path = os.path.join(args.log_dir, f'{save_name}_l2_feature.npy')
    targets_save_path = os.path.join(args.log_dir, f'{save_name}_target.npy')
    mask_save_path = os.path.join(args.log_dir, f'{save_name}_mask.npy')
    np.save(feature_save_path, all_feats)
    np.save(targets_save_path, targets)
    np.save(mask_save_path, mask)


    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    logger_class('Using contrastive feature Done')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    if output_kmeans:
        return all_acc, old_acc, new_acc, kmeans, all_feats
    else:
        return all_acc, old_acc, new_acc




def fake_label_kmeans(model, test_loader,
                epoch, save_name, device,
                args, logger_class=None, K=200):
    model.eval()

    all_feats = []
    all_concat_feats = []
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating features...')

    # First extract all features
    for batch_idx, _item in enumerate(tqdm(test_loader)):
        images = _item[0]
        label = _item[1]
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        # concat_feats = model(images, concat=True)
        feats = model(images, concat=False)
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # all_feats.append(feats.cpu().detach().numpy())
        # targets = np.append(targets, label.cpu().detach().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    logger_class(f'Fitting K-Means... with K={K}, dataset K = {args.num_labeled_classes + args.num_unlabeled_classes}')


    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    logger_class('Using contrastive feature Done')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)


    return all_acc, old_acc, new_acc, kmeans, all_feats




'''Direct test from Memory Buffer Accuracies: All 0.6406 | Old 0.6037 | New 0.6590
0.6405693950177936
0.6037358238825884
0.6589923256589924
==> Create pseudo labels for unlabeled data
[Time] [faiss-gpu] build index 15 consumes 0.0763 s
[Time] [faiss-gpu] query topk 15 consumes 0.0245 s
100%|██████████| 5994/5994 [00:00<00:00, 8545.50it/s]
  0%|          | 0/83359 [00:00<?, ?it/s][Time] get links consumes 0.7021 s
100%|██████████| 83359/83359 [00:00<00:00, 788058.05it/s]
=======================================================
  Infomap v2.6.0 starts at 2022-11-10 16:36:59
  -> Input network: 
  -> No file output!
  -> Configuration: two-level
                    directed
=======================================================
  OpenMP 201511 detected with 10 threads...
  -> Ordinary network input, using the Map Equation for first order network flows
Calculating global network flow using flow model 'directed'... 
  -> Using unrecorded teleportation to links. 
  -> PageRank calculation done in 200 iterations.

  => Sum node flow: 1, sum link flow: 1
Build internal network with 5979 nodes and 83359 links...
  -> Max node flow: 0.000971
  -> Max node in/out degree: 76/14
  -> Max node entropy: 3.807354922
  -> Entropy rate: 3.803799691
  -> One-level codelength: 11.9497309

================================================
Trial 1/1 starting at 2022-11-10 16:36:59
================================================
Two-level compression: 61% 0.0019% 0.04157645% 
Partitioned to codelength 0.00451318608 + 4.68615572 = 4.690668906 in 176 (174 non-trivial) modules.

=> Trial 1/1 finished in 0.065085041s with codelength 4.69066891


================================================
Summary after 1 trial
================================================
Best end modular solution in 2 levels:
Per level number of modules:         [        176,           0] (sum: 176)
Per level number of leaf nodes:      [          0,        5979] (sum: 5979)
Per level average child degree:      [        176,     33.9716] (average: 38.0328)
Per level codelength for modules:    [0.004513186, 0.000000000] (sum: 0.004513186)
Per level codelength for leaf nodes: [0.000000000, 4.686155720] (sum: 4.686155720)
Per level codelength total:          [0.004513186, 4.686155720] (sum: 4.690668906)

===================================================
  Infomap ends at 2022-11-10 16:36:59
  (Elapsed time: 0.111647983s)
===================================================
孤立点数：15
总节点数：5994
总类别数：191/173
cluster cost time: 0.9473400115966797
Collating real label...
100%|██████████| 47/47 [00:05<00:00,  7.90it/s]
Direct test from Memory Buffer Accuracies: All 0.6643 | Old 0.7958 | New 0.5329
0.664330997664331
0.7957957957957958
0.5328661995328662

Process finished with exit code 0
'''