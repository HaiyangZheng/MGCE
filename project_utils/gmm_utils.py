import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

def re_sample(tensor_means, tensor_vars=None, times_of_rsampling=10):
    if tensor_vars is not None:
        dist = MultivariateNormal(tensor_means, tensor_vars)
    else:
        if len(tensor_means.size()) > 1:
            cov = torch.eye(tensor_means.size()[1])
        else:
            cov = torch.eye(tensor_means.size()[0])
        dist = MultivariateNormal(tensor_means, cov)
    results = dist.sample_n(times_of_rsampling)

    return results


class LGMLoss(nn.Module):
    """
    Refer to paper:
    Weitao Wan, Yuanyi Zhong,Tianpeng Li, Jiansheng Chen
    Rethinking Feature Distribution for Loss Functions in Image Classification. CVPR 2018
    re-implement by yirong mao
    2018 07/02
    """
    def __init__(self, num_classes, feat_dim, alpha):
        super(LGMLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        self.log_covs = nn.Parameter(torch.zeros(num_classes, feat_dim), requires_grad=False)

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        log_covs = torch.unsqueeze(self.log_covs, dim=0)


        covs = torch.exp(log_covs) # 1*c*d
        tcovs = covs.repeat(batch_size, 1, 1) # n*c*d
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1) #eq.(18)


        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.alpha)
        y_onehot = y_onehot + 1.0
        margin_dist = torch.mul(dist, y_onehot)

        slog_covs = torch.sum(log_covs, dim=-1) #1*c
        tslog_covs = slog_covs.repeat(batch_size, 1)
        margin_logits = -0.5*(tslog_covs + margin_dist) #eq.(17)
        margin_logits = F.log_softmax(margin_logits, dim=1)

        logits = -0.5 * (tslog_covs + dist)
        logits = F.log_softmax(logits, dim=1)

        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        cdist = cdiff.pow(2).sum(1).sum(0) / 2.0

        slog_covs = torch.squeeze(slog_covs)
        batch_det = torch.sum(torch.index_select(slog_covs, dim=0, index=label.long()))
        reg = 0.5 * (batch_det + 1e-8)
        # likelihood = (1.0/batch_size) * (cdist + reg)
        likelihood = (1.0/batch_size) * (cdist - reg)

        # covs = torch.exp(log_covs)  # 1*c*d
        # covs = torch.squeeze(covs)
        # var_batch = torch.index_select(covs, dim=0, index=label.long())
        # reg2 = 0.5 * torch.log(torch.prod(var_batch))

        return logits, margin_logits, likelihood, cdist, reg

from torch.nn.init import trunc_normal_

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x



class LGMLoss_with_init(nn.Module):
    """
    Refer to paper:
    Weitao Wan, Yuanyi Zhong,Tianpeng Li, Jiansheng Chen
    Rethinking Feature Distribution for Loss Functions in Image Classification. CVPR 2018
    re-implement by yirong mao
    2018 07/02
    """
    def __init__(self, num_classes, feat_dim, hyper_args, initial_means=None, initial_vars=None, project_head=False):
        super(LGMLoss_with_init, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = hyper_args.gmm_alpha
        self.hyper_args = hyper_args
        self.use_project_head = project_head
        if self.use_project_head:
            self.project_head = DINOHead(in_dim=self.feat_dim,
                                         out_dim=self.feat_dim,
                                         nlayers=1)
        else:
            self.project_head = None
        if initial_means is not None:
            _initial_centers = initial_means.detach().clone().to(hyper_args.device)
            self.centers = nn.Parameter(_initial_centers, requires_grad=True)
        else:
            assert False, 'Please use the version without initialization'
        if initial_vars is not None:
            _initial_vars = initial_vars.detach().clone().to(hyper_args.device)
            _initial_log_vars = torch.log(torch.pow(_initial_vars, 2))
            self.log_covs = nn.Parameter(_initial_log_vars, requires_grad=True)
        else:
            self.log_covs = nn.Parameter(torch.zeros(num_classes, feat_dim), requires_grad=False)

    def forward(self, feat, label):
        if self.use_project_head:
            feat = self.project_head(feat)

        batch_size = feat.shape[0]
        log_covs = torch.unsqueeze(self.log_covs, dim=0)


        covs = torch.exp(log_covs) # 1*c*d
        tcovs = covs.repeat(batch_size, 1, 1) # n*c*d
        diff = torch.unsqueeze(feat, dim=1) - torch.unsqueeze(self.centers, dim=0)
        wdiff = torch.div(diff, tcovs)
        diff = torch.mul(diff, wdiff)
        dist = torch.sum(diff, dim=-1) #eq.(18)


        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).to(self.hyper_args.device)
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.alpha)
        y_onehot = y_onehot + 1.0
        margin_dist = torch.mul(dist, y_onehot)

        slog_covs = torch.sum(log_covs, dim=-1) #1*c
        tslog_covs = slog_covs.repeat(batch_size, 1)
        margin_logits = -0.5*(tslog_covs + margin_dist) #eq.(17)
        margin_logits = F.log_softmax(margin_logits, dim=1)

        logits = -0.5 * (tslog_covs + dist)
        logits = F.log_softmax(logits, dim=1)

        cdiff = feat - torch.index_select(self.centers, dim=0, index=label.long())
        cdist = cdiff.pow(2).sum(1).sum(0) / 2.0

        slog_covs = torch.squeeze(slog_covs)
        batch_det = torch.sum(torch.index_select(slog_covs, dim=0, index=label.long()))
        reg = 0.5 * (batch_det + 1e-8)
        # likelihood = (1.0/batch_size) * (cdist + reg)
        likelihood = (1.0/batch_size) * (cdist - reg)

        # covs = torch.exp(log_covs)  # 1*c*d
        # covs = torch.squeeze(covs)
        # var_batch = torch.index_select(covs, dim=0, index=label.long())
        # reg2 = 0.5 * torch.log(torch.prod(var_batch))

        return logits, margin_logits, likelihood, cdist, reg


