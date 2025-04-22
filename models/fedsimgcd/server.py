import time
import wandb
import numpy as np
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from models.gcn_e import GCN_E
from misc.utils import *
from models.nets import *
from modules.federated import ServerModule
from copy import deepcopy
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch
from project_utils.cluster_evaluate_utils import log_accs_from_preds
from torch.utils.data import DataLoader

class Server(ServerModule):
    def __init__(self, args, sd, gpu_server, dataset_dict):
        super(Server, self).__init__(args, sd, gpu_server)
        self.model = GCN(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(self.gpu_id)
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
        self.args.mlp_out_dim = self.args.num_labeled_classes + self.args.num_unlabeled_classes
        # projector = DINOHead(in_dim=self.args.feat_dim, out_dim=self.args.mlp_out_dim, nlayers=self.args.num_mlp_layers)
        # model = nn.Sequential(backbone, projector).cuda(g_id)
        self.backbone = backbone.cuda(self.gpu_id)

    def on_round_begin(self, curr_rnd): #server on a round ->
        self.round_begin = time.time()
        print('test before agg')
        self.curr_rnd = curr_rnd
        self.test_by_kmeans(curr_rnd)
        self.sd['global'] = self.get_weights()

    def on_round_complete(self, updated, curr_rnd=None):
        self.update(updated)
        self.save_state()
        print('test after agg')
        self.test_by_kmeans(curr_rnd)

    def test_by_kmeans(self, curr_rnd):
        if curr_rnd is not None:
            pass
        if self.curr_rnd % self.args.test_interval == 0:
            seen_class_list = list(self.dataset_dict[f'client-0-global_testing'].information['global seen classes'])
            hyperparameter_K = self.dataset_dict[f'client-0-global_testing'].information['num global classes']
            test_dataset = deepcopy(self.dataset_dict[f'client-0-global_testing'])
            test_loader = DataLoader(test_dataset, num_workers=0, batch_size=self.args.batch_size, shuffle=False)
            all_acc, old_acc, new_acc = self.test_kmeans(backbone=self.backbone, test_loader=test_loader,epoch=curr_rnd,
                                                         seen_class_list=seen_class_list, hyperparameter_K=hyperparameter_K,
                                                         args=self.args)
            wandb.log(
                {f"Generalization All": all_acc, f"Generalization Old": old_acc, f"Generalization New": new_acc})
            print('Generalization Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,new_acc))



    def update(self, updated):
        st = time.time()
        model_local_weights = []
        backbone_local_weights = []
        local_train_sizes = []
        for c_id in updated:
            backbone_local_weights.append(self.sd[c_id]['backbone'].copy())
            model_local_weights.append(self.sd[c_id]['model'].copy())
            local_train_sizes.append(self.sd[c_id]['train_size'])
            del self.sd[c_id]
        self.logger.print(f'all clients have been uploaded ({time.time()-st:.2f}s)')

        st = time.time()
        ratio = (np.array(local_train_sizes)/np.sum(local_train_sizes)).tolist()

        self.set_weights(self.model, self.aggregate(model_local_weights, ratio))
        self.set_weights(self.backbone, self.aggregate(backbone_local_weights, ratio))
        self.logger.print(f'global model has been updated ({time.time()-st:.2f}s)')

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model),
            'backbone': get_state_dict(self.backbone)
        }

    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
            'backbone': get_state_dict(self.backbone),
        })

    def test_kmeans(self, backbone, test_loader,
                    epoch, seen_class_list, hyperparameter_K,
                    args, save_name='global_generalization'):
        backbone.eval()

        all_feats = []
        targets = np.array([])
        mask = np.array([])
        print('Collating features... for generalization evaluation')

        # First extract all features
        for batch_idx, _item in enumerate(tqdm(test_loader)):
            images = _item[0]
            label = _item[1]
            images = images.cuda(self.gpu_id)

            # Pass features through base model and then additional learnable transform (linear layer)
            # concat_feats = model(images, concat=True)

            feats = backbone(images)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.detach().cpu().numpy())
            targets = np.append(targets, label.detach().cpu().numpy())

            mask = np.append(mask, np.array([True if x.item() in seen_class_list
                                             else False for x in label]))


        all_feats = np.concatenate(all_feats)


        print(f'Using class token features Fitting K-Means... k={hyperparameter_K}')
        kmeans = KMeans(n_clusters=hyperparameter_K, random_state=0).fit(all_feats)
        preds = kmeans.labels_

        print('Using class token features Done')

        # -----------------------
        # EVALUATE
        # -----------------------
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                        T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                        writer=None)


        return all_acc, old_acc, new_acc






