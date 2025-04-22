from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
import numpy as np


class CM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features.half()) # for fp16
            # grad_inputs = grad_outputs.mm(ctx.features.half()) # for fp16

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, targets, features, momentum=0.5, device=None):
    if device is None:
        return CM.apply(inputs, targets, features.to(inputs.device), torch.Tensor([momentum]).to(inputs.device))
    else:
        return CM.apply(inputs.cuda(device), targets.cuda(device), features.cuda(device), torch.Tensor([momentum]).cuda(device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples=None, temp=0.05, momentum=0.2,
                 use_hard=False, cluster=None, use_sym=False, num_instances=16, gpu_id=None):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.clustering_results = cluster
        self.use_hard = use_hard
        self.device = gpu_id
        self.use_sym = use_sym
        # self.device = 'cuda'
        self.num_instances = num_instances
        if num_samples is not None:
            self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets, center_weights=None,
                kmeans_centeroids=None,
                kmeans_pids=None, neg_size=20, indexes=None,
                use_nmi=False):
        inputs = F.normalize(inputs, dim=1).cuda(self.device)
        targets = targets.cuda(self.device)
        outputs = cm(inputs, targets, self.features, self.momentum, self.device)
        # cm is  128 101  probility of input belong to 101 prototype
        # assign weights for each ID
        if center_weights is not None:
            batch_weights = center_weights[targets]
            loss = (F.cross_entropy(outputs / self.temp, targets, reduce='none') * batch_weights).mean()
            return loss

        if kmeans_centeroids is not None:
            # epoch 1 in cub
            # inputs 128*768    kmeans_centeroids 101 * 768
            finch_score = outputs  # not devided by temp
            finch_pos = finch_score.gather(1, targets.view(-1, 1)) # find right class similarity
            # finding neg in kmeans clustering
            kmeans_score = inputs @ kmeans_centeroids.cuda(self.device).t()
            temp_score_neg = kmeans_score.detach().clone()

            # kmeans pids for current batch
            batch_kmeans_pids = kmeans_pids[indexes].long()

            # generate confidence mask
            confidence_mask, batch_conf = [], []
            for idx in range(inputs.shape[0] // (2 * self.num_instances)):
                id_range_head = idx * self.num_instances
                id_range_rear = (1 + idx) * self.num_instances
                kmeans_pids = batch_kmeans_pids[id_range_head:id_range_rear].cpu().numpy()
                # find cluster PID
                max_count = np.bincount(kmeans_pids).argmax()
                confidence_mask.extend((kmeans_pids == max_count).tolist())
            confidence_mask = torch.tensor(confidence_mask * 2).float().cuda(self.device)  # replicate

            # find hard nega
            temp_score_neg.scatter_(1, batch_kmeans_pids.view(-1, 1), -2)  # exclude pos
            sel_ind_neg = torch.sort(temp_score_neg, dim=1)[1][:, -neg_size:]
            # choose hard neg response from KMeans clustering
            kmeans_chosen_neg = kmeans_score.gather(1, sel_ind_neg)

            mixed_score = torch.cat([finch_pos, kmeans_chosen_neg], 1) / self.temp
            total_mask = torch.zeros((mixed_score.shape[0])).cuda(self.device).long()
            neg_con_loss = (confidence_mask * F.cross_entropy(mixed_score, total_mask, reduction='none')).mean()
            # return (confidence_mask*F.cross_entropy(outputs / self.temp, targets, reduction='none')).mean()
            if use_nmi:
                bsize = confidence_mask.shape[0] // 2
                batch_conf = confidence_mask[:bsize]
                return 0.2 * neg_con_loss + F.cross_entropy(outputs / self.temp, targets), batch_conf

            return 0.2 * neg_con_loss + F.cross_entropy(outputs / self.temp, targets)

        return F.cross_entropy(outputs / self.temp, targets)