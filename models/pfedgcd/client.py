import time
import torch
import torch.nn.functional as F

from misc.utils import *
from models.nets import *
from modules.federated import ClientModule

class Client(ClientModule):

    def __init__(self, args, w_id, g_id, sd):
        super(Client, self).__init__(args, w_id, g_id, sd)
        self.model = MaskedGCN(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args.l1, self.args).cuda(g_id) 
        self.parameters = list(self.model.parameters())

    def init_state(self):
        # #### initialize GCD model
        # from data.get_datasets import get_datasets, get_class_splits
        # self.args = get_class_splits(self.args)
        #
        # self.args.num_labeled_classes = len(self.args.train_classes)
        # self.args.num_unlabeled_classes = len(self.args.unlabeled_classes)
        #
        # self.args.logger.info(f'Using evaluation function {self.args.eval_funcs[0]} to print results')
        #
        # # ----------------------
        # # BASE MODEL
        # # ----------------------
        # self.args.interpolation = 3
        # self.args.crop_pct = 0.875
        #
        # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        #
        # if self.args.warmup_model_dir is not None:
        #     self.args.logger.info(f'Loading weights from {self.args.warmup_model_dir}')
        #     backbone.load_state_dict(torch.load(self.args.warmup_model_dir, map_location='cpu'))
        #
        # # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        # self.args.image_size = 224
        # self.args.feat_dim = 768
        # self.args.num_mlp_layers = 3
        # self.args.mlp_out_dim = self.args.num_labeled_classes + self.args.num_unlabeled_classes
        #
        # # ----------------------
        # # HOW MUCH OF BASE MODEL TO FINETUNE
        # # ----------------------
        # for m in backbone.parameters():
        #     m.requires_grad = False
        #
        # # Only finetune layers from block 'args.grad_from_block' onwards
        # for name, m in backbone.named_parameters():
        #     if 'block' in name:
        #         block_num = int(name.split('.')[1])
        #         if block_num >= self.args.grad_from_block:
        #             m.requires_grad = True
        #
        # self.args.logger.info('model build')
        #
        # # --------------------
        # # CONTRASTIVE TRANSFORM
        # # --------------------
        # train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
        # train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
        # # --------------------
        # # DATASETS
        # # --------------------
        # train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
        #                                                                                      train_transform,
        #                                                                                      test_transform,
        #                                                                                      args)
        #
        # # --------------------
        # # SAMPLER
        # # Sampler which balances labelled and unlabelled examples in each batch
        # # --------------------
        # label_len = len(train_dataset.labelled_dataset)
        # unlabelled_len = len(train_dataset.unlabelled_dataset)
        # sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
        # sample_weights = torch.DoubleTensor(sample_weights)
        # sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))
        #
        # # --------------------
        # # DATALOADERS
        # # --------------------
        # train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
        #                           sampler=sampler, drop_last=True, pin_memory=True)
        # test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
        #                                     batch_size=256, shuffle=False, pin_memory=False)
        # # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
        # #                                   batch_size=256, shuffle=False, pin_memory=False)
        #
        # # ----------------------
        # # PROJECTION HEAD
        # # ----------------------
        # projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
        # model = nn.Sequential(backbone, projector).to(device)
        #
        # # ----------------------
        # # TRAIN
        # # ----------------------
        # # train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
        # train(model, train_loader, None, test_loader_unlabelled, args)

        self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.base_lr, weight_decay=self.args.weight_decay)
        self.log = {
            'lr': [],'train_lss': [],
            'ep_local_val_lss': [],'ep_local_val_acc': [],
            'rnd_local_val_lss': [],'rnd_local_val_acc': [],
            'ep_local_test_lss': [],'ep_local_test_acc': [],
            'rnd_local_test_lss': [],'rnd_local_test_acc': [],
            'rnd_sparsity':[], 'ep_sparsity':[]
        }

    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),
            'log': self.log,
        })

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.log = loaded['log']
    
    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd
        self.update(self.sd[f'personalized_{self.client_id}' \
            if (f'personalized_{self.client_id}' in self.sd) else 'global'])
        self.global_w = convert_np_to_tensor(self.sd['global']['model'], self.gpu_id)

    def update(self, update):
        self.prev_w = convert_np_to_tensor(update['model'], self.gpu_id)
        set_state_dict(self.model, update['model'], self.gpu_id, skip_stat=True, skip_mask=True)

    def on_round_begin(self):
        self.train()
        self.transfer_to_server()

    def get_sparsity(self):
        n_active, n_total = 0, 1
        for mask in self.masks:
            pruned = torch.abs(mask) < self.args.l1
            mask = torch.ones(mask.shape).cuda(self.gpu_id).masked_fill(pruned, 0)
            n_active += torch.sum(mask)
            _n_total = 1
            for s in mask.shape:
                _n_total *= s 
            n_total += _n_total
        return ((n_total-n_active)/n_total).item()

    def train(self):
        st = time.time()
        val_local_acc, val_local_lss = self.validate(mode='valid')
        test_local_acc, test_local_lss = self.validate(mode='test')
        self.logger.print(
            f'rnd: {self.curr_rnd+1}, ep: {0}, '
            f'val_local_loss: {val_local_lss.item():.4f}, val_local_acc: {val_local_acc:.4f}, lr: {self.get_lr()} ({time.time()-st:.2f}s)'
        )
        self.log['ep_local_val_acc'].append(val_local_acc)
        self.log['ep_local_val_lss'].append(val_local_lss)
        self.log['ep_local_test_acc'].append(test_local_acc)
        self.log['ep_local_test_lss'].append(test_local_lss)

        self.masks = []
        for name, param in self.model.state_dict().items():
            if 'mask' in name: self.masks.append(param) 

        for ep in range(self.args.n_eps):
            st = time.time()
            self.model.train()
            for _, batch in enumerate(self.loader.pa_loader):
                self.optimizer.zero_grad()
                batch = batch.cuda(self.gpu_id)
                y_hat = self.model(batch)
                train_lss = F.cross_entropy(y_hat[batch.train_mask], batch.y[batch.train_mask])
                
                #################################################################
                for name, param in self.model.state_dict().items():
                    if 'mask' in name:
                        train_lss += torch.norm(param.float(), 1) * self.args.l1
                    elif 'conv' in name or 'clsif' in name:
                        if self.curr_rnd == 0: continue
                        train_lss += torch.norm(param.float()-self.prev_w[name], 2) * self.args.loc_l2
                #################################################################
                        
                train_lss.backward()
                self.optimizer.step()
            
            sparsity = self.get_sparsity()
            val_local_acc, val_local_lss = self.validate(mode='valid')
            test_local_acc, test_local_lss = self.validate(mode='test')
            self.logger.print(
                f'rnd:{self.curr_rnd+1}, ep:{ep+1}, '
                f'val_local_loss: {val_local_lss.item():.4f}, val_local_acc: {val_local_acc:.4f}, lr: {self.get_lr()} ({time.time()-st:.2f}s)'
            )
            self.log['train_lss'].append(train_lss.item())
            self.log['ep_local_val_acc'].append(val_local_acc)
            self.log['ep_local_val_lss'].append(val_local_lss)
            self.log['ep_local_test_acc'].append(test_local_acc)
            self.log['ep_local_test_lss'].append(test_local_lss)
            self.log['ep_sparsity'].append(sparsity)
        self.log['rnd_local_val_acc'].append(val_local_acc)
        self.log['rnd_local_val_lss'].append(val_local_lss)
        self.log['rnd_local_test_acc'].append(test_local_acc)
        self.log['rnd_local_test_lss'].append(test_local_lss)
        self.log['rnd_sparsity'].append(sparsity)
        self.save_log()

    @torch.no_grad()
    def get_functional_embedding(self):
        self.model.eval()
        with torch.no_grad():
            proxy_in = self.sd['proxy']
            proxy_in = proxy_in.cuda(self.gpu_id)
            proxy_out = self.model(proxy_in, is_proxy=True)
            proxy_out = proxy_out.mean(dim=0)
            proxy_out = proxy_out.clone().detach().cpu().numpy()
        return proxy_out

    def transfer_to_server(self):
        self.sd[self.client_id] = {
            'model': get_state_dict(self.model),
            'train_size': len(self.loader.partition),
            'functional_embedding': self.get_functional_embedding()
        }




    
    
