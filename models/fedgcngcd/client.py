import time
import wandb
import torch
import torch.nn.functional as F
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, \
    get_params_groups
from torch.optim import SGD, lr_scheduler
from util.general_utils import AverageMeter, init_experiment
import math
from misc.utils import *
from models.nets import *
from modules.federated import ClientModule
from copy import deepcopy
from util.cluster_and_log_utils import log_accs_from_preds
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from models.gcn_e import GCN_E
from data.utils.gcn_clustering import train_gcn, test_gcn_e
from project_utils.contrastive_utils import extract_features
from data.gcn_e_dataset import GCNEDataset
from project_utils.infomap_cluster_utils import generate_cluster_features
from project_utils.gcn_cluster_memory_utils import ClusterMemory
from sklearn.cluster import KMeans
from project_utils.sampler import RandomMultipleGallerySamplerNoCam
from project_utils.data_utils import IterLoader, FakeLabelDataset
from torch.utils.data import DataLoader


class Client(ClientModule):

    def __init__(self, args, w_id, g_id, sd, dataset_dict):
        super(Client, self).__init__(args, w_id, g_id, sd)
        # self.model = GCN(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(g_id)
        self.model = GCN_E(feature_dim=args.feat_dim, nhid=512, nclass=2)
        self.parameters = list(self.model.parameters())
        self.dataset_dict = deepcopy(dataset_dict)

        # switch -> init_state or load_state   in federated

        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in backbone.parameters():
            m.requires_grad = False

        # Only finetune layers from block '_args.grad_from_block' onwards
        for name, m in backbone.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= self.args.grad_from_block:
                    m.requires_grad = True
        # model = nn.Sequential(backbone, projector).cuda(g_id)
        self.backbone = backbone.cuda(g_id)

    def init_state(self):
        # test on unlabel train set [1] = num unlabeled classes    [4] label and unlabel   [0] label but [0]+[1]>[4]
        # print(f'client-{self.client_id}-global_testing')
        # print(self.dataset_dict[f'client-{self.client_id}-global_testing'].information[f'client-{self.client_id}'])
        self.mlp_out_dim = \
        self.dataset_dict[f'client-{self.client_id}-global_testing'].information[f'client-{self.client_id}'][4]
        # self.projector = DINOHead(in_dim=self.args.feat_dim, out_dim=self.args.mlp_out_dim, nlayers=self.args.num_mlp_layers)
        self.projector = DINOHead(in_dim=self.args.n_feat, out_dim=self.mlp_out_dim, nlayers=self.args.num_mlp_layers)
        self.projector = self.projector.cuda(self.gpu_id)

        student = nn.Sequential(self.backbone, self.projector)
        params_groups = get_params_groups(student)
        self.optimizer = SGD(params_groups, lr=self.args.lr, momentum=self.args.momentum,
                             weight_decay=self.args.weight_decay)
        # self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.base_lr, weight_decay=self.args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.n_rnds,
            eta_min=self.args.lr * 1e-3,
        )
        if self.args.warmup_opt:

            self.exp_lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1,
                                                           total_epoch=self.args.num_warmup_epoch,
                                                           after_scheduler=exp_lr_scheduler)
        else:
            self.exp_lr_scheduler = exp_lr_scheduler

        self.cluster_criterion = DistillLoss(
            self.args.warmup_teacher_temp_epochs,
            self.args.n_rnds,
            self.args.n_views,
            self.args.warmup_teacher_temp,
            self.args.teacher_temp,
        )
        self.fp16_scaler = None
        if self.args.fp16:
            self.fp16_scaler = torch.cuda.amp.GradScaler()
        # # inductive
        # best_test_acc_lab = 0
        # # transductive
        # best_train_acc_lab = 0
        # best_train_acc_ubl = 0
        # best_train_acc_all = 0

        self.log = {
            'lr': [], 'train_lss': [],
            'ep_local_test_all_acc': [], 'ep_local_test_old_acc': [],
            'ep_local_test_new_acc': [], 'rnd_local_test_all_acc': [],
            'rnd_local_test_old_acc': [], 'rnd_local_test_new_acc': [],

        }

    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),
            'backbone': get_state_dict(self.backbone),
            'projector': get_state_dict(self.projector),
            'log': self.log,
        })

    def load_state(self):
        if self.last_client_id == -1:
            loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        else:
            loaded = torch_load(self.args.checkpt_path, f'{self.last_client_id}_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        set_state_dict(self.backbone, loaded['backbone'], self.gpu_id)
        set_state_dict(self.projector, loaded['projector'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.log = loaded['log']

    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd
        self.update(self.sd['global'])

    def update(self, update):
        set_state_dict(self.model, update['model'], self.gpu_id, skip_stat=True)
        set_state_dict(self.backbone, update['backbone'], self.gpu_id, skip_stat=True)
        # set_state_dict(self.projector, update['projector'], self.gpu_id, skip_stat=True)

    def on_round_begin(self):
        self.train()
        self.transfer_to_server()

    def train_graph(self):

        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            # including train and test data
            cluster_loader = deepcopy(self.loaders_dict['clustering'])
            features, labels, if_labeled = extract_features(
                self.backbone, cluster_loader, print_freq=50, device=self.gpu_id
            )
            features = torch.cat(features, dim=0)  # Vit features
            label_mark = torch.cat(labels, dim=0)  # real labels for labelled samples
            if_labeled = torch.cat(if_labeled, dim=0)  # whether it is unlabeled or labeled samples

            # split labelled
            features_arr = F.normalize(features, p=2, dim=1).cpu().numpy()
            un_features = F.normalize(features[~if_labeled, :], p=2, dim=1).cpu().numpy()
            lab_features = F.normalize(features[if_labeled, :], p=2, dim=1).cpu().numpy()
            lab_pids, unlab_pids = label_mark[if_labeled], label_mark[~if_labeled]

        # form training sets based on features
        gcn_e_dataset = GCNEDataset(
            features=lab_features, labels=lab_pids, knn=self.args.k1,
            feature_dim=768, is_norm=True, th_sim=0,
            max_conn=self.args.max_conn, conf_metric=False,
            ignore_ratio=0.1, ignore_small_confs=True,
            use_candidate_set=True, radius=0.3
        )
        # train the gcns based on current features
        # self.model.train()
        gcn_e = train_gcn(self.model, gcn_e_dataset, self.args.gcn_train_epoch)
        # use GCN to predict edge of unlabeled samples
        gcn_e_test = GCNEDataset(
            features=features_arr, labels=label_mark, knn=self.args.k1,
            feature_dim=768, is_norm=True, th_sim=0,
            max_conn=self.args.max_conn, conf_metric=False,
            ignore_ratio=0.7, ignore_small_confs=True,
            use_candidate_set=True, radius=0.3
        )
        pseudo_labels = test_gcn_e(
            gcn_e, gcn_e_test, if_labelled=if_labeled,
            train_pid_count=int(lab_pids.max()) + 1,
            max_conn=self.args.max_conn
        )
        num_cluster = len(np.unique(pseudo_labels))
        print(f"Predicted pids: {num_cluster}, Real pids: {int(1 + label_mark.max())}")

        # Create hybrid memory
        num_fea = self.backbone.module.num_features if isinstance(self.backbone,
                                                                  nn.DataParallel) else self.backbone.num_features
        # memory for computing loss based on hyri clustering results
        self.memory = None
        memory = ClusterMemory(
            num_fea, num_cluster, temp=self.args.temp,
            momentum=self.args.memory_momentum, use_sym=self.args.use_sym,
            num_instances=self.args.num_instances, gpu_id=self.gpu_id
        ).cuda(self.gpu_id)
        cluster_features = generate_cluster_features(pseudo_labels, features)
        finch_centers = F.normalize(cluster_features, dim=1).cuda(self.gpu_id)
        memory.features = finch_centers

        self.memory = memory

        pseudo_labeled_dataset = []

        # generate kmeans results based on finch centers
        kfunc = KMeans(n_clusters=num_cluster, init=cluster_features.numpy())
        kmeans_pseudo_labels = kfunc.fit_predict(features_arr)

        # import faiss
        # niter = 300
        # d = features_arr.shape[1]
        # kfunc = faiss.Kmeans(d, num_cluster, niter=niter, gpu=True)
        # kfunc.train(features_arr)
        # kmeans_pseudo_labels = kfunc.index.search(x=features_arr, k=1)[1].reshape(-1)

        # instance-level
        self.kmeans_centeroids = F.normalize(
            generate_cluster_features(kmeans_pseudo_labels, features),
            p=2, dim=1).cuda(self.gpu_id)
        self.kmeans_pseudo_labels = torch.from_numpy(kmeans_pseudo_labels).cuda(self.gpu_id)

        # building dataset with unlabelled data
        # self.loaders_dict[f'client-{self.client_id}-fake'] = None
        self.contrastive_cluster_train_loader = None

        for i, (_item, label) in enumerate(
                zip(self.dataset_dict[f'client-{self.client_id}-clustering'].data, pseudo_labels)):
            if isinstance(_item, str):
                pseudo_labeled_dataset.append((_item, label.item(), i))
            elif isinstance(_item, (list, tuple, np.ndarray)):
                if isinstance(_item[0], str):
                    pseudo_labeled_dataset.append((_item[0], label.item(), i))
                else:
                    pseudo_labeled_dataset.append((_item[1], label.item(), i))
        train_transform = self.dataset_dict[f'client-{self.client_id}-training'].transform
        PK_sampler = RandomMultipleGallerySamplerNoCam(pseudo_labeled_dataset, self.args.num_instances)
        contrastive_cluster_train_loader = IterLoader(
            DataLoader(
                FakeLabelDataset(pseudo_labeled_dataset, root=None, transform=train_transform),
                batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                sampler=PK_sampler, shuffle=False, pin_memory=True, drop_last=True
            )
        )
        contrastive_cluster_train_loader.new_epoch()
        # self.loaders_dict[f'client-{self.client_id}-fake'] = contrastive_cluster_train_loader
        self.contrastive_cluster_train_loader = contrastive_cluster_train_loader

    def train(self):
        if self.args.contrastive_graph_weight > 0:
            self.train_graph()

        self.backbone.train()
        self.projector.train()
        self.save_log()
        all_acc, old_acc, new_acc = 0.0, 0.0, 0.0
        for ep in range(self.args.n_eps):
            st = time.time()
            train_lss = self.train_one_epoch(epoch=self.curr_rnd)
            # val_local_acc, val_local_lss = self.validate(mode='valid')
            # test_local_acc, test_local_lss = self.validate(mode='test')
            if self.curr_rnd % self.args.test_interval == 0:
                all_acc, old_acc, new_acc = self.test_by_projector(epoch=self.curr_rnd)

            self.logger.print(
                f'rnd:{self.curr_rnd + 1}, ep:{ep + 1}, '
                f'test_local_all_acc: {all_acc:.4f}, test_local_old_acc: {old_acc:.4f}, test_local_new_acc: {new_acc:.4f}, lr: {self.get_lr()} ({time.time() - st:.2f}s)'
            )
            self.log['train_lss'].append(train_lss)

            self.log['ep_local_test_all_acc'].append(all_acc)
            self.log['ep_local_test_old_acc'].append(old_acc)
            self.log['ep_local_test_new_acc'].append(new_acc)

        wandb.log(
            {f"{self.client_id}-all": all_acc, f"{self.client_id}-old": old_acc, f"{self.client_id}-new": new_acc})
        self.log['rnd_local_test_all_acc'].append(all_acc)
        self.log['rnd_local_test_old_acc'].append(old_acc)
        self.log['rnd_local_test_new_acc'].append(new_acc)
        self.save_log()

    def transfer_to_server(self):
        self.sd[self.client_id] = {
            'model': get_state_dict(self.model),
            'backbone': get_state_dict(self.backbone),
            'projector': get_state_dict(self.projector),
            'train_size': len(self.loaders_dict['training'])
        }

    def train_one_epoch(self, epoch):
        loss_record = AverageMeter()
        for batch_idx, batch in enumerate(self.loaders_dict['training']):
            images, class_labels, uq_idxs, mask_lab, attrabute = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(self.gpu_id, non_blocking=True), mask_lab.cuda(self.gpu_id,
                                                                                                      non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(self.gpu_id, non_blocking=True)

            with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                class_token_output = self.backbone(images)
                student_proj, student_out = self.projector(class_token_output)
                teacher_out = student_out.detach()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = self.cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += self.args.memax_weight * me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj, device=self.gpu_id)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels, device=self.gpu_id)

                graph_images, graph_pids, graph_indexes = self.contrastive_cluster_train_loader.next()
                graph_indexes = graph_indexes.cuda(self.gpu_id)

                graph_images = torch.cat(graph_images, dim=0).cuda(self.gpu_id)
                graph_index_dup, graph_pids_dup = graph_indexes.detach().clone(), graph_pids.clone()
                graph_indexes = torch.cat((graph_indexes, graph_index_dup), dim=0).cuda(self.gpu_id)
                graph_pids = torch.cat((graph_pids, graph_pids_dup), dim=0).cuda(self.gpu_id)
                # graph_f_out = self.backbone(graph_images)[0]
                graph_f_out = self.backbone(graph_images)
                loss = 0
                if self.args.contrastive_graph_weight > 0:
                    contrastive_cluster_loss = self.memory(
                        graph_f_out, graph_pids, None, self.kmeans_centeroids, self.kmeans_pseudo_labels,
                        self.args.neg_size, graph_indexes
                    )
                    loss += self.args.contrastive_graph_weight * contrastive_cluster_loss.cuda(self.gpu_id)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                pstr += f'contrastive_graph_loss: {contrastive_cluster_loss.item():.4f} '

                loss += (1 - self.args.sup_weight) * cluster_loss + self.args.sup_weight * cls_loss
                loss += (1 - self.args.sup_weight) * contrastive_loss + self.args.sup_weight * sup_con_loss

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            self.optimizer.zero_grad()
            if self.fp16_scaler is None:
                loss.backward()
                self.optimizer.step()
            else:
                self.fp16_scaler.scale(loss).backward()
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()

            if batch_idx % self.args.print_freq == 0:
                self.logger.print('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                                  .format(epoch, batch_idx, len(self.loaders_dict['training']), loss.item(), pstr))

        self.logger.print('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))
        # Step schedule
        self.exp_lr_scheduler.step()
        return

    @torch.no_grad()
    def test_by_projector(self, epoch):
        self.logger.print('Testing on unlabelled examples in the training data...')
        self.model.eval()
        self.backbone.eval()
        self.projector.eval()
        preds, targets = [], []
        mask = np.array([])
        for batch_idx, _item in enumerate(self.loaders_dict['local_testing']):
            images = _item[0].cuda(self.gpu_id, non_blocking=True)
            label = _item[1]
            with torch.no_grad():
                class_token_output = self.backbone(images)
                _, logits = self.projector(class_token_output)
                preds.append(logits.argmax(1).cpu().numpy())
                targets.append(label.cpu().numpy())
                mask = np.append(mask, np.array(
                    [True if x.item() in range(len(self.args.train_classes)) else False for x in label]))

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                        T=epoch, eval_funcs=self.args.eval_funcs,
                                                        save_name='Train ACC Unlabelled',
                                                        args=self.args)

        self.logger.print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

        return all_acc, old_acc, new_acc
