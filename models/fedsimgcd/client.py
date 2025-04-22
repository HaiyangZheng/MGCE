import time
import wandb
import torch
import torch.nn.functional as F
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
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

class Client(ClientModule):

    def __init__(self, args, w_id, g_id, sd, dataset_dict):
        super(Client, self).__init__(args, w_id, g_id, sd)
        self.model = GCN(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(g_id)
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
        self.mlp_out_dim = self.dataset_dict[f'client-{self.client_id}-global_testing'].information[f'client-{self.client_id}'][4]
        # self.projector = DINOHead(in_dim=self.args.feat_dim, out_dim=self.args.mlp_out_dim, nlayers=self.args.num_mlp_layers)
        self.projector = DINOHead(in_dim=self.args.n_feat, out_dim=self.mlp_out_dim, nlayers=self.args.num_mlp_layers)
        self.projector = self.projector.cuda(self.gpu_id)
        student = nn.Sequential(self.backbone, self.projector)
        params_groups = get_params_groups(student)
        self.optimizer = SGD(params_groups, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
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
            'lr': [],'train_lss': [],
            'ep_local_test_all_acc': [],'ep_local_test_old_acc': [],
            'ep_local_test_new_acc': [],'rnd_local_test_all_acc': [],
            'rnd_local_test_old_acc': [],'rnd_local_test_new_acc': [],

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
        # if self.last_client_id == -1:
        #     loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        # else:
        #     loaded = torch_load(self.args.checkpt_path, f'{self.last_client_id}_state.pt')
        loaded_client = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        loaded_server = torch_load(self.args.checkpt_path, f'server_state.pt')
        set_state_dict(self.model, loaded_server['model'], self.gpu_id)
        set_state_dict(self.backbone, loaded_server['backbone'], self.gpu_id)
        set_state_dict(self.projector, loaded_client['projector'], self.gpu_id)
        self.optimizer.load_state_dict(loaded_client['optimizer'])
        self.log = loaded_client['log']
    
    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd
        self.update(self.sd['global'])

    def update(self, update):
        set_state_dict(self.model, update['model'], self.gpu_id, skip_stat=False)
        set_state_dict(self.backbone, update['backbone'], self.gpu_id, skip_stat=False)
        # set_state_dict(self.projector, update['projector'], self.gpu_id, skip_stat=True)


    def on_round_begin(self):
        self.train()
        self.transfer_to_server()

    def train(self):
        self.model.train()
        self.backbone.train()
        self.projector.train()
        self.save_log()
        for ep in range(self.args.n_eps):
            st = time.time()
            train_lss = self.train_one_epoch(epoch=self.curr_rnd)
            # val_local_acc, val_local_lss = self.validate(mode='valid')
            # test_local_acc, test_local_lss = self.validate(mode='test')
            all_acc, old_acc, new_acc = self.test_by_projector(epoch=self.curr_rnd)

            self.logger.print(
                f'rnd:{self.curr_rnd + 1}, ep:{ep + 1}, '
                f'test_local_all_acc: {all_acc:.4f}, test_local_old_acc: {old_acc:.4f}, test_local_new_acc: {new_acc:.4f}, lr: {self.get_lr()} ({time.time() - st:.2f}s)'
            )
            self.log['train_lss'].append(train_lss)

            self.log['ep_local_test_all_acc'].append(all_acc)
            self.log['ep_local_test_old_acc'].append(old_acc)
            self.log['ep_local_test_new_acc'].append(new_acc)

        wandb.log({f"{self.client_id}-all": all_acc, f"{self.client_id}-old": old_acc, f"{self.client_id}-new":new_acc})
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

            class_labels, mask_lab = class_labels.cuda(self.gpu_id, non_blocking=True), mask_lab.cuda(self.gpu_id, non_blocking=True).bool()
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

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss = 0
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
        masks, preds, targets = [], [], []
        # mask = np.array([])
        for batch_idx, _item in enumerate(self.loaders_dict['local_testing']):
            images = _item[0].cuda(self.gpu_id, non_blocking=True)
            label = _item[1]
            mask = _item[-1]

            with torch.no_grad():
                class_token_output = self.backbone(images)
                _, logits = self.projector(class_token_output)
                preds.append(logits.argmax(1).cpu().numpy())
                targets.append(label.cpu().numpy())
                masks.append(mask.cpu().numpy())
                # mask = np.append(mask, np.array(
                #     [True if x.item() in range(len(self.args.train_classes)) else False for x in label]))

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        masks = np.concatenate(masks)
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=masks,
                                                        T=epoch, eval_funcs=self.args.eval_funcs, save_name='Train ACC Unlabelled',
                                                        args=self.args)

        self.logger.print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))


        return all_acc, old_acc, new_acc
