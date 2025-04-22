import time
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F

from .cluster_utils import AverageMeter


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def extract_features3(model, data_loader, print_freq=50, device=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = []
    labels = []
    indexs = []

    end = time.time()
    with torch.no_grad():
        for i, _item in enumerate(data_loader):
            imgs = _item[0]
            targets = _item[1]
            # uq_idx = _item[2]
            if_train = _item[3][:, 0].bool()
            data_time.update(time.time() - end)
            if device is not None:
                imgs = to_torch(imgs).cuda(device)
            else:
                imgs = to_torch(imgs).cuda()
            outputs = model(imgs)
            outputs = outputs.data.cpu()

            features.append(outputs)
            labels.append(targets)
            # indexs.append(uq_idx)
            indexs.append(if_train)


            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels, indexs


def extract_features(model, data_loader, print_freq=50, feature_output_index=0,args=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = []
    labels = []
    indexs = []

    end = time.time()
    with torch.no_grad():
        for i, _item in enumerate(data_loader):
            imgs = _item[0]
            targets = _item[1]
            # uq_idx = _item[2]
            if_train = _item[3][:, 0].bool()
            data_time.update(time.time() - end)
            if args is not None:
                imgs = to_torch(imgs).to(args.device)
            else:
                imgs = to_torch(imgs).cuda()
            if feature_output_index == -1:
                outputs = model(imgs)
                outputs = outputs.data.cpu()
            else:
                outputs = model(imgs)
                outputs = outputs[feature_output_index].data.cpu()

            features.append(outputs)
            labels.append(targets)
            # indexs.append(uq_idx)
            indexs.append(if_train)


            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels, indexs



def extract_features4UFGGCD(backbone, projector, data_loader, name, args):
    backbone.eval()
    projector.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # results_dict = defaultdict(list)
    temp_results_dict = dict()


    temp_results_dict[name + "_features"] = []
    temp_results_dict[name + "_labels"] = []
    temp_results_dict[name + "_if_labeded"] = []
    # temp_results_dict[name + "_if_old_class"] = []

    end = time.time()
    with torch.no_grad():
        for i, _item in enumerate(data_loader):
            imgs = _item[0]
            targets = _item[1]
            # uq_idx = _item[2]
            if_train = _item[3][:, 0].bool()
            # if_old_class = _item[4][:, 0].bool()
            data_time.update(time.time() - end)
            imgs = to_torch(imgs).to(args.device)
            # features = [features[:, 0, :], features[:, 1:1 + num_prompts, :]]
            outputs = backbone(imgs)

            proj_out = projector(outputs[name])
            proj_cluster_contrastive_outputs = proj_out[args.feature_output_index].data.cpu()
            temp_results_dict[name + "_features"].append(proj_cluster_contrastive_outputs)
            temp_results_dict[name + "_labels"].append(targets)
            temp_results_dict[name + "_if_labeded"].append(if_train)
            # temp_results_dict[name + "_if_old_class"].append(if_old_class)

            batch_time.update(time.time() - end)
            end = time.time()
        results = dict()

        results[name + "_features"] = torch.cat(temp_results_dict[name + "_features"], dim=0)
        results[name + "_labels"] = torch.cat(temp_results_dict[name + "_labels"], dim=0)
        results[name + "_if_labeded"] = torch.cat(temp_results_dict[name + "_if_labeded"], dim=0)
        # results[name + "_if_old_class"] = torch.cat(temp_results_dict[name + "_if_old_class"], dim=0)

        return results

def extract_features_with_headlist(backbone, project_dict, data_loader, args):
    backbone.eval()
    project_dict.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # results_dict = defaultdict(list)
    temp_results_dict = dict()
    for name in project_dict.keys():
        temp_results_dict[name + "_features"] = []
        temp_results_dict[name + "_labels"] = []
        temp_results_dict[name + "_if_labeded"] = []
        # temp_results_dict[name + "_if_old_class"] = []

    ###added by haiyang for backbone feature test
    temp_results_dict['backbone' + "_features"] = []
    temp_results_dict['backbone' + "_labels"] = []
    temp_results_dict['backbone' + "_if_labeded"] = []
    # temp_results_dict['backbone' + "_if_old_class"] = []  

    end = time.time()
    with torch.no_grad():
        for i, _item in enumerate(data_loader):
            imgs = _item[0]
            targets = _item[1]
            # uq_idx = _item[2]
            if_train = _item[3][:, 0].bool()
            # if_old_class = _item[4][:, 0].bool()
            data_time.update(time.time() - end)
            imgs = to_torch(imgs).to(args.device)
            # features = [features[:, 0, :], features[:, 1:1 + num_prompts, :]]
            outputs = backbone(imgs, True)

            for i_, (name, projector) in enumerate(project_dict.items()):
                # # 对应visual prompt
                # proj_out = projector(outputs[:, i_, :])
                # 没有visual prompt的cls token
                proj_out = projector(outputs[:, 0, :])
                # args.feature_output_index=2 返回cls token
                proj_cluster_contrastive_outputs = proj_out[args.feature_output_index].data.cpu()
                # proj_cluster_contrastive_outputs = proj_out[0].data.cpu()
                temp_results_dict[name + "_features"].append(proj_cluster_contrastive_outputs)
                temp_results_dict[name + "_labels"].append(targets)
                temp_results_dict[name + "_if_labeded"].append(if_train)
                # temp_results_dict[name + "_if_old_class"].append(if_old_class)

            ###added by haiyang for backbone feature tes
            backbone_features = outputs[:, 0, :].data.cpu()
            temp_results_dict['backbone' + "_features"].append(backbone_features)
            temp_results_dict['backbone' + "_labels"].append(targets)
            temp_results_dict['backbone' + "_if_labeded"].append(if_train)
            # temp_results_dict['backbone' + "_if_old_class"].append(if_old_class)        

            batch_time.update(time.time() - end)
            end = time.time()
        results = dict()
        for name in project_dict.keys():
            results[name + "_features"] = torch.cat(temp_results_dict[name + "_features"], dim=0)
            results[name + "_labels"] = torch.cat(temp_results_dict[name + "_labels"], dim=0)
            results[name + "_if_labeded"] = torch.cat(temp_results_dict[name + "_if_labeded"], dim=0)
            # results[name + "_if_old_class"] = torch.cat(temp_results_dict[name + "_if_old_class"], dim=0)

        ###added by haiyang for backbone feature test
        results['backbone' + "_features"] = torch.cat(temp_results_dict['backbone' + "_features"], dim=0)
        results['backbone' + "_labels"] = torch.cat(temp_results_dict['backbone' + "_labels"], dim=0)
        results['backbone' + "_if_labeded"] = torch.cat(temp_results_dict['backbone' + "_if_labeded"], dim=0)
        # results['backbone' + "_if_old_class"] = torch.cat(temp_results_dict['backbone' + "_if_old_class"], dim=0)

        return results
    
## added by haiyang为了测试两阶段模型修改的特征提取代码，不影响之前文件
def extract_features_with_headlist_twostage(backbone, project_dict, data_loader, args):
    backbone.eval()
    project_dict.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # results_dict = defaultdict(list)
    temp_results_dict = dict()
    for name in project_dict.keys():
        temp_results_dict[name + "_features"] = []
        temp_results_dict[name + "_labels"] = []
        temp_results_dict[name + "_if_labeded"] = []
        # temp_results_dict[name + "_if_old_class"] = []

    ###added by haiyang for backbone feature test
    temp_results_dict['backbone' + "_features"] = []
    temp_results_dict['backbone' + "_labels"] = []
    temp_results_dict['backbone' + "_if_labeded"] = []
    # temp_results_dict['backbone' + "_if_old_class"] = []  

    end = time.time()
    with torch.no_grad():
        for i, _item in enumerate(data_loader):
            imgs = _item[0]
            targets = _item[1]
            # uq_idx = _item[2]
            if_train = _item[3][:, 0].bool()
            # if_old_class = _item[4][:, 0].bool()
            data_time.update(time.time() - end)
            imgs = to_torch(imgs).to(args.device)
            # features = [features[:, 0, :], features[:, 1:1 + num_prompts, :]]
            outputs = backbone(imgs)

            for i_, (name, projector) in enumerate(project_dict.items()):
                # # 对应visual prompt
                # proj_out = projector(outputs[:, i_, :])
                # 没有visual prompt的cls token
                proj_out = projector(outputs)
                # args.feature_output_index=2 返回cls token
                proj_cluster_contrastive_outputs = proj_out[args.feature_output_index].data.cpu()
                # proj_cluster_contrastive_outputs = proj_out[0].data.cpu()
                temp_results_dict[name + "_features"].append(proj_cluster_contrastive_outputs)
                temp_results_dict[name + "_labels"].append(targets)
                temp_results_dict[name + "_if_labeded"].append(if_train)
                # temp_results_dict[name + "_if_old_class"].append(if_old_class)

            ###added by haiyang for backbone feature tes
            backbone_features = outputs.data.cpu()
            temp_results_dict['backbone' + "_features"].append(backbone_features)
            temp_results_dict['backbone' + "_labels"].append(targets)
            temp_results_dict['backbone' + "_if_labeded"].append(if_train)
            # temp_results_dict['backbone' + "_if_old_class"].append(if_old_class)        

            batch_time.update(time.time() - end)
            end = time.time()
        results = dict()
        for name in project_dict.keys():
            results[name + "_features"] = torch.cat(temp_results_dict[name + "_features"], dim=0)
            results[name + "_labels"] = torch.cat(temp_results_dict[name + "_labels"], dim=0)
            results[name + "_if_labeded"] = torch.cat(temp_results_dict[name + "_if_labeded"], dim=0)
            # results[name + "_if_old_class"] = torch.cat(temp_results_dict[name + "_if_old_class"], dim=0)

        ###added by haiyang for backbone feature test
        results['backbone' + "_features"] = torch.cat(temp_results_dict['backbone' + "_features"], dim=0)
        results['backbone' + "_labels"] = torch.cat(temp_results_dict['backbone' + "_labels"], dim=0)
        results['backbone' + "_if_labeded"] = torch.cat(temp_results_dict['backbone' + "_if_labeded"], dim=0)
        # results['backbone' + "_if_old_class"] = torch.cat(temp_results_dict['backbone' + "_if_old_class"], dim=0)

        return results

def extract_features2(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, targets, uq_idx, attribute) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(uq_idx, outputs, targets):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [(correct[:k].reshape(-1).float().sum(0) * 100. / batch_size).cpu().numpy() for k in topk]


def info_nce_logits(features, args):
    b_ = 0.5 * int(features.size(0))  # features:[B*2/expert_num] , b_: B/expert_num

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)  # [2*b_]
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [2*b_, 2*b_]
    labels = labels.to(args.device)

    if args.negative_mixup:
        labels = torch.cat([torch.arange(b_ * 2) for i in range(args.n_views)], dim=0)  # [4*b_]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # [4*b_, 4*b_]
        labels = labels.to(args.device)

        beta = np.random.beta(0.2, 0.2)
        feat_idx = torch.arange(features.shape[0] - 1, -1, -1)
        inter_feat = beta * features.detach().clone() + (1 - beta) * features[feat_idx].detach().clone()
        inter_feat = F.normalize(inter_feat, dim=1)  # [b_*2, c]
        features = torch.cat([inter_feat, features])

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

    logits = logits / args.temperature
    return logits, labels


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, device, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
