import numpy as np
from sklearn.cluster import KMeans
import torch
from project_utils.cluster_and_log_utils import log_accs_from_preds

from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from methods.estimate_k.estimate_k import scipy_optimise, binary_search

from data.gcn_e_dataset import GCNEDataset
from data.utils import test_gcn_e

# TODO: Debug
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch.nn as nn

@torch.no_grad()
def test_kmeans_bs(
    model, test_loader, epoch, save_name, device, 
    args, logger_class=None, in_training=False, max_classes=250
):
    K = binary_search(test_loader, args, max_classes, model)
    logger_class(f"The best K for testing {K}")
    model.eval()
    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])  # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])  # From all the data, which instances belong to Old classes

    # First extract all features
    for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(test_loader):
        images = images.to(device)
        with torch.no_grad():
            feats = model(images)
        if args.use_l2_in_ssk:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                    else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)

    l_feats = all_feats[mask_lab]  # Get labelled set
    u_feats = all_feats[~mask_lab]  # Get unlabelled set
    l_targets = targets[mask_lab]  # Get labelled targets
    u_targets = targets[~mask_lab]  # Get unlabelled targets
    kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, 
                           init='k-means++', n_init=args.k_means_init, random_state=None, 
                           n_jobs=None, pairwise_batch_size=512, mode=None)
    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for x in (l_feats, u_feats, l_targets, u_targets))
    kmeans.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeans.labels_.cpu().numpy()
    u_targets = u_targets.cpu().numpy()
    preds = all_preds[~mask_lab]

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)
    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=u_targets, y_pred=preds, mask=mask, 
        eval_funcs=args.eval_funcs, save_name=save_name, T=epoch, print_output=True
    )

    if args.use_contrastive_cluster and args.contrastive_cluster_method == 'ssk':
        return all_acc, old_acc, new_acc, kmeans, all_feats
    else:
        return all_acc, old_acc, new_acc, kmeans, None

@torch.no_grad()
def test_kmeans_scipy(
    model, test_loader, epoch, save_name, device, 
    args, logger_class=None, in_training=False, max_classes=100
):
    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """
    K = scipy_optimise(test_loader, args, max_classes, model)
    logger_class(f"The best K for testing {K}")
    
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
        for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(test_loader):
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
        # not based on 
        kmeans = SemiSupKMeans(
            k=K, tolerance=1e-4, max_iterations=max_kmeans_iter, 
            init='k-means++', n_init=args.k_means_init, random_state=None, 
            n_jobs=None, pairwise_batch_size=512, mode=None
        )

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
        for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(test_loader):
            images = images.to(device)
            with torch.no_grad():
                feats = model(images)
            if args.use_l2_in_ssk:
                feats = torch.nn.functional.normalize(feats, dim=-1)
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

        l_feats = all_feats[mask_lab]  # Get labelled set
        u_feats = all_feats[~mask_lab]  # Get unlabelled set
        l_targets = targets[mask_lab]  # Get labelled targets
        u_targets = targets[~mask_lab]  # Get unlabelled targets
        if in_training:
            max_kmeans_iter = args.train_max_kmeans_iter
        else:
            max_kmeans_iter = args.max_kmeans_iter
        logger_class(f'Using estimated K for K-Means... max_kmeans_iter = {max_kmeans_iter}')
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
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask,
                                                        eval_funcs=args.eval_funcs,
                                                        save_name=save_name, T=epoch, print_output=True)

        if args.use_contrastive_cluster and args.contrastive_cluster_method == 'ssk':
            return all_acc, old_acc, new_acc, kmeans, all_feats
        else:
            return all_acc, old_acc, new_acc, kmeans, None

@torch.no_grad()
def test_kmeans_semi_sup(model, test_loader, epoch, save_name, device, args, K=None, logger_class=None, in_training=False):
    if K is None:
        K = args.num_labeled_classes + args.num_unlabeled_classes
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
        for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(test_loader):
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
        # not based on 
        kmeans = SemiSupKMeans(
            k=K, tolerance=1e-4, max_iterations=max_kmeans_iter, 
            init='k-means++', n_init=args.k_means_init, random_state=None, 
            n_jobs=None, pairwise_batch_size=512, mode=None
        )

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
        for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(test_loader):
            images = images.to(device)
            feats = model(images)
            if args.use_l2_in_ssk:
                feats = torch.nn.functional.normalize(feats, dim=-1)
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

        l_feats = all_feats[mask_lab]  # Get labelled set
        u_feats = all_feats[~mask_lab]  # Get unlabelled set
        l_targets = targets[mask_lab]  # Get labelled targets
        u_targets = targets[~mask_lab]  # Get unlabelled targets
        if in_training:
            max_kmeans_iter = args.train_max_kmeans_iter
        else:
            max_kmeans_iter = args.max_kmeans_iter
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
        all_acc, old_acc, new_acc = log_accs_from_preds(
            y_true=u_targets, y_pred=preds, mask=mask, 
            eval_funcs=args.eval_funcs, save_name=save_name, 
            T=epoch, print_output=True
        )

        if args.use_contrastive_cluster and args.contrastive_cluster_method == 'ssk':
            return all_acc, old_acc, new_acc, kmeans, all_feats
        else:
            return all_acc, old_acc, new_acc, kmeans, None


@torch.no_grad()
def test_gcn_semi(
    model, gcn_e, test_loader, epoch, save_name, device, args
):
    model.eval()
    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])  # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])  # From all the data, which instances belong to Old classes

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(test_loader):
        images = images.to(device)
        feats = model(images)
        if args.use_l2_in_ssk:
            feats = torch.nn.functional.normalize(feats, dim=-1)
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

    l_feats = all_feats[mask_lab]  # Get labelled set
    u_feats = all_feats[~mask_lab]  # Get unlabelled set
    l_targets = targets[mask_lab]  # Get labelled targets
    u_targets = targets[~mask_lab]  # Get unlabelled targets
    import ipdb;ipdb.set_trace()
    
    gcn_e.eval()
    
    gcn_e_test = GCNEDataset(
        features=all_feats, labels=targets, knn=args.k1, 
        feature_dim=768, is_norm=True, th_sim=0, 
        max_conn=args.max_conn, conf_metric=False, 
        ignore_ratio=0.7, ignore_small_confs=True, 
        use_candidate_set=True, radius=0.3
    )
    pseudo_labels = test_gcn_e(
        gcn_e, gcn_e_test, if_labelled=mask_lab, 
        train_pid_count=int(l_targets.max())+1,
        max_conn=args.max_conn
    )
    
    # kmeans.fit_mix(u_feats, l_feats, l_targets)
    all_preds = pseudo_labels.cpu().numpy()
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
    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=u_targets, y_pred=preds, mask=mask, 
        eval_funcs=args.eval_funcs, save_name=save_name, 
        T=epoch, print_output=True
    )

    if args.use_contrastive_cluster and args.contrastive_cluster_method == 'ssk':
        return all_acc, old_acc, new_acc, kmeans, all_feats
    else:
        return all_acc, old_acc, new_acc, kmeans, None


@torch.no_grad()
def test_kmeans(model, test_loader,
                epoch, save_name, device,
                args, logger_class=None, 
                output_kmeans=False, num_clusters=None):
    model.eval()

    all_feats = []
    all_concat_feats = []
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating features...')

    # First extract all features
    for batch_idx, _item in enumerate(test_loader):
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
        # for True or False, True for seen labels while False for unseen labels
        mask = np.append(
            mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label])
        )

    logger_class('Using no_l2 feature Fitting K-Means...')
    all_concat_feats = np.concatenate(all_concat_feats)

    # clustering all features
    num_clusters = args.num_labeled_classes + args.num_unlabeled_classes if num_clusters is None else num_clusters
    kmeans = KMeans(
        n_clusters=num_clusters, random_state=0
    ).fit(all_concat_feats)
    preds = kmeans.labels_
    logger_class('Using no_l2 feature Done')
    
    # import faiss
    # n_clusters=args.num_labeled_classes + args.num_unlabeled_classes
    # niter = 300
    # d = all_concat_feats.shape[1]
    # kfunc = faiss.Kmeans(d, n_clusters, niter=niter, gpu=True)
    # kfunc.train(all_concat_feats)
    # # cluster_centers_ = kfunc.centroids
    # preds_faiss = kfunc.index.search(x=all_concat_feats, k=1)[1].reshape(-1)
    
    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=mask, T=epoch, 
        eval_funcs=args.eval_funcs, save_name=save_name + 'add'
    )
    logger_class('no_l2 feature kmeans Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))

    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    logger_class('Using contrastive feature Done')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=mask, T=epoch, 
        eval_funcs=args.eval_funcs, save_name=save_name
    )
    if output_kmeans:
        return all_acc, old_acc, new_acc, kmeans, all_feats
    else:
        return all_acc, old_acc, new_acc




@torch.no_grad()
def test_kmeans2(model, test_loader,
                epoch, save_name, device,
                args, logger_class=None, 
                output_kmeans=False, num_clusters=None):
    model.eval()

    all_feats = []
    all_concat_feats = []
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating features...')

    # First extract all features
    for batch_idx, _item in enumerate(test_loader):
        images = _item[0]
        label = _item[1]
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        # concat_feats = model(images, concat=True)
        feats = model(images)
        concat_feats = feats.detach().clone()
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_concat_feats.append(concat_feats.cpu().numpy())
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # for True or False, True for seen labels while False for unseen labels
        mask = np.append(
            mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label])
        )

    logger_class('Using no_l2 feature Fitting K-Means...')
    all_concat_feats = np.concatenate(all_concat_feats)

    # clustering all features
    num_clusters = args.num_labeled_classes + args.num_unlabeled_classes if num_clusters is None else num_clusters
    kmeans = KMeans(
        n_clusters=num_clusters, random_state=0
    ).fit(all_concat_feats)
    preds = kmeans.labels_
    logger_class('Using no_l2 feature Done')
    
    # import faiss
    # n_clusters=args.num_labeled_classes + args.num_unlabeled_classes
    # niter = 300
    # d = all_concat_feats.shape[1]
    # kfunc = faiss.Kmeans(d, n_clusters, niter=niter, gpu=True)
    # kfunc.train(all_concat_feats)
    # # cluster_centers_ = kfunc.centroids
    # preds_faiss = kfunc.index.search(x=all_concat_feats, k=1)[1].reshape(-1)
    
    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=mask, T=epoch, 
        eval_funcs=args.eval_funcs, save_name=save_name + 'add'
    )
    logger_class('no_l2 feature kmeans Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))

    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    logger_class('Using contrastive feature Done')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=mask, T=epoch, 
        eval_funcs=args.eval_funcs, save_name=save_name
    )
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
    for batch_idx, _item in enumerate(test_loader):
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
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name)


    return all_acc, old_acc, new_acc, kmeans, all_feats




