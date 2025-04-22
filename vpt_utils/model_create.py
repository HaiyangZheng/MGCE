import torch
import torch.nn as nn
from . import vision_transformer as vits
from . import vpt_vision_transformer as vpt_vit


def create_backbone(args, device):
    """ create ViT backbone
    the process includes following steps:
    1. load DINO state_dict
    2. load pretrained state_dict
    3. init ViT/VPT-ViT with loaded state_dict
    4. freeze backbone parameters
    """
    model = vits.__dict__['vit_base']()
    state_dict = torch.load(args.dino_pretrain_path, map_location='cpu')
    model.load_state_dict(state_dict)
    if args.warmup_model_dir is not None:
        print(f'Loading weights from {args.warmup_model_dir}')
        model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    ### VPT
    if args.use_vpt:
        if args.unfrezee_only_last_prompt:
            vptmodel = vpt_vit.__dict__['vit_base_last_prompt'](
                num_prompts=args.num_prompts,
                vpt_dropout=args.vpt_dropout,
                n_shallow_prompts=args.n_shallow_prompts,
            )
            if args.load_from_model is not None:
                print(f'NOTE:: load from {args.load_from_model}')
                # vptmodel.load_state_dict(torch.load(args.load_from_model, map_location='cpu'), strict=True)
                vptmodel.load_state_dict(torch.load(args.load_from_model, map_location='cpu'), strict=False)
            else:
                vptmodel.load_from_state_dict(state_dict, False)
            model = vptmodel
            vpt_vit.configure_parameters_last_token(model=model,
                                                    grad_layer=args.grad_from_block)
        else:
            vptmodel = vpt_vit.__dict__['vit_base'](
                num_prompts=args.num_prompts,
                vpt_dropout=args.vpt_dropout,
                n_shallow_prompts=args.n_shallow_prompts,
            )
            if args.load_from_model is not None:
                print(f'NOTE:: load from {args.load_from_model}')
                vptmodel.load_state_dict(torch.load(args.load_from_model, map_location='cpu'), strict=True)
            else:
                vptmodel.load_from_state_dict(state_dict, False)
            model = vptmodel

            vpt_vit.configure_parameters(model=model,
                                         grad_layer=args.grad_from_block)  ### configure parameters [freeze/unfreeze]

    else:
        ### ViT
        for m in model.parameters():
            m.requires_grad = False
        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
    model.to(device)
    return model

## added by haiyang 不影响其他文件
class DINOHead_generator_new(nn.Module):
    def __init__(self, out_dim, norm_last_layer=True, hidden_dim=1536):
        super().__init__()

        self.mlp = nn.Linear(768, hidden_dim)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(hidden_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x_proj = nn.ReLU()(x_proj)

        # x = nn.functional.normalize(x_proj, dim=-1, p=2)
        # logits = self.last_layer(x)
        return x_proj

## added by haiyang 不影响其他文件
def create_backbone_twostage(args, device):
    """ create ViT backbone
    the process includes following steps:
    1. load DINO state_dict
    2. load pretrained state_dict
    3. init ViT/VPT-ViT with loaded state_dict
    4. freeze backbone parameters
    """
    # lr01_epochs50
    known_classes_pretrain_model = '/leonardo_work/IscrC_Fed-GCD/hyzheng/ConceptGCD/exp/stage1/iscap_stage1_lr01_epochs50_(28.12.2024_|_43.619)/checkpoints/model.pt'
    backbone = vits.__dict__['vit_base']()

    checkpoint = torch.load(known_classes_pretrain_model, map_location='cpu')
    new_state_dict = {}
    for key, value in checkpoint['model'].items():
        if key.startswith('0.'):
            new_key = key[2:]
            new_state_dict[new_key] = value

    backbone.load_state_dict(new_state_dict)


    for m in backbone.parameters():
        m.requires_grad = False

    projector = DINOHead_generator_new(out_dim=args.mlp_out_dim, hidden_dim=768)
    model = nn.Sequential(backbone, projector)

    model.to(device)
    return model

def create_projection_head(args, device, use_checkpoint=True):
    """ create projection heads (with state_dict)"""
    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                                                out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projection_head.to(device)
    if (args.load_from_head is not None) and (use_checkpoint == True):
        print(f'NOTE: load head from {args.load_from_head}')
        projection_head.load_state_dict(torch.load(args.load_from_head, map_location='cpu'), strict=True)
    return projection_head


def create_model(args, device):
    """create backbone and projection head (with state_dict)"""
    model = create_backbone(args, device)
    projection_head = create_projection_head(args, device)
    return [model, projection_head]


def create_dino_backbone(args, dino_pretrain_path, device, arch='vit_base'):
    """ create DINO backbone with state_dict"""
    model = vits.__dict__[arch]()
    state_dict = torch.load(dino_pretrain_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    for m in model.parameters():
        m.requires_grad = False
    model.to(device)
    return model
