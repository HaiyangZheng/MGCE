import torch
import torch.nn as nn
from .vision_transformer import *

import numpy as np

from functools import reduce
from operator import mul
from .scale_forward import forward as multiscale_forward

class VPT_ViT(VisionTransformer):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3,
                 num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 embed_layer=PatchEmbed,
                 num_prompts=1,
                 vpt_dropout=0.0,
                 n_shallow_prompts=0, **kwargs):

        # Recreate ViT
        super(VPT_ViT, self).__init__(img_size, patch_size, in_chans, num_classes,
                                      embed_dim, depth, num_heads, mlp_ratio,
                                      qkv_bias, qk_scale,
                                      drop_rate, attn_drop_rate, drop_path_rate,
                                      norm_layer, **kwargs)
        print('NOTE: to check the arguments of the model creation factory!')
        ### initialize prompts
        self.num_prompts = num_prompts
        self.n_shallow_prompts = n_shallow_prompts

        assert self.n_shallow_prompts < self.num_prompts

        self.prompt_tokens = nn.Parameter(torch.zeros(depth, self.num_prompts, embed_dim))
        # frozen shallow tokens
        self.last_prompt_tokens = nn.Parameter(torch.zeros(1, self.num_prompts, embed_dim))

        self.vpt_drop = nn.ModuleList([nn.Dropout(p=vpt_dropout) for d in range(depth)])

        ### re-initialize positional embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + self.num_prompts, embed_dim))

        trunc_normal_(self.prompt_tokens, std=.02)
        trunc_normal_(self.last_prompt_tokens, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        with torch.no_grad():
            self.mask_vpt_pos_embed()

        return

    def mask_vpt_pos_embed(self):
        self.pos_embed[:, 1:self.num_prompts + 1, :] = 0.0
        return

    def unfreeze_prompt(self):
        for m in self.parameters():
            m.requires_grad = False
        self.prompt_tokens.requires_grad = True
        return


    def load_from_state_dict(self, state_dict, strict=False):
        """ load state_dict from DINO pre-trained model
        """
        init_weight = self.pos_embed.data
        pos_embed = state_dict.pop('pos_embed')  # manual loading
        init_weight[0, 0, :] = pos_embed[0, 0, :]
        init_weight[0, 1 + self.num_prompts:, :] = pos_embed[0, 1:, :]
        self.pos_embed.data = init_weight
        self.load_state_dict(state_dict, strict=strict)
        return

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1 - self.num_prompts
        N = self.pos_embed.shape[1] - 1 - self.num_prompts
        if npatch == N and w == h:
            return self.pos_embed

        ### TODO: test, corrected
        class_pos_embed = self.pos_embed[:, :1 + self.num_prompts, :]
        patch_pos_embed = self.pos_embed[:, 1 + self.num_prompts:, :]
        # print(f'class_pos_embed={class_pos_embed.shape} patch_pos_embed={patch_pos_embed.shape}')
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # B, L, D

        # add the [CLS] and [PROMPT] token to the embed patch tokens
        cls_token = self.cls_token.expand(B, -1, -1)
        prompt_tokens = self.prompt_tokens[0].expand(B, -1, -1)
        prompt_tokens = self.vpt_drop[0](prompt_tokens)
        x = torch.cat((cls_token, prompt_tokens, x), dim=1)  # B, 1+P+L, D

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x, return_all_patches=False):
        x = self.prepare_tokens(x)  # B, L, D
        B = x.size(0)
        n_vpt_layer = self.prompt_tokens.size(0) - 1
        for idx_layer, blk in enumerate(self.blocks):
            x = blk(x)
            if idx_layer < n_vpt_layer:
                ### exclude precedent prompts
                a = x[:, 0, :].unsqueeze(1) if self.n_shallow_prompts == 0 else x[:, :1 + self.n_shallow_prompts, :]
                c = x[:, self.num_prompts + 1:, :]
                ### generate prompt input
                b = self.prompt_tokens[idx_layer + 1, self.n_shallow_prompts:, :].expand(B, -1,
                                                                                         -1)  # corrected by i+1, origical i
                b = self.vpt_drop[idx_layer + 1](b)
                x = torch.cat([a, b, c], dim=1)
        x = self.norm(x)
        if return_all_patches:
            return x
        else:
            return x[:, 0]

    def get_last_vpt_selfattention(self, x):
        assert self.n_shallow_prompts == 0
        x = self.prepare_tokens(x)
        B = x.size(0)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
                ### exclude precedent prompts
                a = x[:, 0, :].unsqueeze(1)
                c = x[:, self.num_prompts + 1:, :]
                b = self.prompt_tokens[i, :, :].expand(B, -1, -1)
                x = torch.cat([a, b, c], dim=1)
            else:
                # return attention of the last block
                x, attn = blk(x, return_attention=True)
                x = self.norm(x)
                return x, attn

    def get_intermediate_layers(self, x):
        assert self.n_shallow_prompts == 0
        x = self.prepare_tokens(x)
        B = x.size(0)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            output.append(self.norm(x))
            ### exclude precedent prompts
            a = x[:, 0, :].unsqueeze(1)
            c = x[:, self.num_prompts + 1:, :]
            b = self.prompt_tokens[i, :, :].expand(B, -1, -1)
            x = torch.cat([a, b, c], dim=1)
        output = torch.stack(output, dim=1)
        return output

def vit_base_last_prompt_multiscale(patch_size=16, num_prompts=1, **kwargs):
    model = VPT_ViT_last_prompt_multiscale(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_prompts=num_prompts,
        **kwargs)
    return model


def vit_base_last_prompt(patch_size=16, num_prompts=1, **kwargs):
    model = VPT_ViT_last_prompt(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_prompts=num_prompts,
        **kwargs)
    return model


def vit_base(patch_size=16, num_prompts=1, **kwargs):
    model = VPT_ViT(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_prompts=num_prompts,
        **kwargs)
    return model


def configure_parameters(model, grad_layer=11):
    model.unfreeze_prompt()

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in model.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= grad_layer:
                m.requires_grad = True
    return

def configure_parameters_last_token(model, grad_layer=11):
    model.unfreeze_last_prompt()

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in model.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= grad_layer:
                m.requires_grad = True
    return

class VPT_ViT_last_prompt(VisionTransformer):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3,
                 num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 embed_layer=PatchEmbed,
                 num_prompts=1,
                 vpt_dropout=0.0,
                 n_shallow_prompts=0, **kwargs):

        # Recreate ViT
        super(VPT_ViT_last_prompt, self).__init__(img_size, patch_size, in_chans, num_classes,
                                      embed_dim, depth, num_heads, mlp_ratio,
                                      qkv_bias, qk_scale,
                                      drop_rate, attn_drop_rate, drop_path_rate,
                                      norm_layer, **kwargs)
        print('NOTE: to check the arguments of the model creation factory!')
        ### initialize prompts
        self.num_prompts = num_prompts
        self.n_shallow_prompts = n_shallow_prompts

        assert self.n_shallow_prompts < self.num_prompts

        self.prompt_tokens = nn.Parameter(torch.zeros(depth, self.num_prompts, embed_dim))
        # frozen shallow tokens
        self.last_prompt_tokens = nn.Parameter(torch.zeros(1, self.num_prompts, embed_dim))

        self.vpt_drop = nn.ModuleList([nn.Dropout(p=vpt_dropout) for d in range(depth)])

        ### re-initialize positional embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + self.num_prompts, embed_dim))

        trunc_normal_(self.prompt_tokens, std=.02)
        trunc_normal_(self.last_prompt_tokens, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        with torch.no_grad():
            self.mask_vpt_pos_embed()

        return

    def mask_vpt_pos_embed(self):
        self.pos_embed[:, 1:self.num_prompts + 1, :] = 0.0
        return

    def unfreeze_prompt(self):
        for m in self.parameters():
            m.requires_grad = False
        self.prompt_tokens.requires_grad = True
        self.last_prompt_tokens.requires_grad = True
        return

    def unfreeze_last_prompt(self):
        for m in self.parameters():
            m.requires_grad = False
        self.last_prompt_tokens.requires_grad = True
        return

    def load_from_state_dict(self, state_dict, strict=False):
        """ load state_dict from DINO pre-trained model
        """
        init_weight = self.pos_embed.data
        pos_embed = state_dict.pop('pos_embed')  # manual loading
        init_weight[0, 0, :] = pos_embed[0, 0, :]
        init_weight[0, 1 + self.num_prompts:, :] = pos_embed[0, 1:, :]
        self.pos_embed.data = init_weight
        self.load_state_dict(state_dict, strict=strict)
        return

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1 - self.num_prompts
        N = self.pos_embed.shape[1] - 1 - self.num_prompts
        if npatch == N and w == h:
            return self.pos_embed

        ### TODO: test, corrected
        class_pos_embed = self.pos_embed[:, :1 + self.num_prompts, :]
        patch_pos_embed = self.pos_embed[:, 1 + self.num_prompts:, :]
        # print(f'class_pos_embed={class_pos_embed.shape} patch_pos_embed={patch_pos_embed.shape}')
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # B, L, D

        # add the [CLS] and [PROMPT] token to the embed patch tokens
        cls_token = self.cls_token.expand(B, -1, -1)
        prompt_tokens = self.prompt_tokens[0].expand(B, -1, -1)
        prompt_tokens = self.vpt_drop[0](prompt_tokens)
        x = torch.cat((cls_token, prompt_tokens, x), dim=1)  # B, 1+P+L, D

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x, return_all_patches=False):
        x = self.prepare_tokens(x)  # B, L, D
        B = x.size(0)
        n_vpt_layer = self.prompt_tokens.size(0) - 1
        for idx_layer, blk in enumerate(self.blocks):
            x = blk(x)
            if idx_layer < n_vpt_layer - 1: # use pretrained frozen prompt
                ### exclude precedent prompts
                a = x[:, 0, :].unsqueeze(1) if self.n_shallow_prompts == 0 else x[:, :1 + self.n_shallow_prompts, :]
                c = x[:, self.num_prompts + 1:, :]
                ### generate prompt input
                b = self.prompt_tokens[idx_layer + 1, self.n_shallow_prompts:, :].expand(B, -1,
                                                                                         -1)  # corrected by i+1, origical i
                b = self.vpt_drop[idx_layer + 1](b)
                x = torch.cat([a, b, c], dim=1)
            elif idx_layer == n_vpt_layer - 1:
                a = x[:, 0, :].unsqueeze(1) if self.n_shallow_prompts == 0 else x[:, :1 + self.n_shallow_prompts, :]
                c = x[:, self.num_prompts + 1:, :]
                ### generate prompt input
                b = self.last_prompt_tokens[0, self.n_shallow_prompts:, :].expand(B, -1,
                                                                                         -1)  # corrected by i+1, origical i
                b = self.vpt_drop[idx_layer + 1](b)
                x = torch.cat([a, b, c], dim=1)

        x = self.norm(x)
        if return_all_patches:
            return x
        else:
            return x[:, 0]

    def get_last_vpt_selfattention(self, x):
        assert self.n_shallow_prompts == 0
        x = self.prepare_tokens(x)
        B = x.size(0)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
                ### exclude precedent prompts
                a = x[:, 0, :].unsqueeze(1)
                c = x[:, self.num_prompts + 1:, :]
                b = self.prompt_tokens[i, :, :].expand(B, -1, -1)
                x = torch.cat([a, b, c], dim=1)
            else:
                # return attention of the last block
                x, attn = blk(x, return_attention=True)
                x = self.norm(x)
                return x, attn

    def get_intermediate_layers(self, x):
        assert self.n_shallow_prompts == 0
        x = self.prepare_tokens(x)
        B = x.size(0)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            output.append(self.norm(x))
            ### exclude precedent prompts
            a = x[:, 0, :].unsqueeze(1)
            c = x[:, self.num_prompts + 1:, :]
            b = self.prompt_tokens[i, :, :].expand(B, -1, -1)
            x = torch.cat([a, b, c], dim=1)
        output = torch.stack(output, dim=1)
        return output






class VPT_ViT_last_prompt_multiscale(VisionTransformer):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3,
                 num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 embed_layer=PatchEmbed,
                 num_prompts=1,
                 vpt_dropout=0.0,
                 n_shallow_prompts=0, **kwargs):

        # Recreate ViT
        super(VPT_ViT_last_prompt_multiscale, self).__init__(img_size, patch_size, in_chans, num_classes,
                                      embed_dim, depth, num_heads, mlp_ratio,
                                      qkv_bias, qk_scale,
                                      drop_rate, attn_drop_rate, drop_path_rate,
                                      norm_layer, **kwargs)
        print('NOTE: to check the arguments of the model creation factory!')
        ### initialize prompts
        self.num_prompts = num_prompts
        self.n_shallow_prompts = n_shallow_prompts

        assert self.n_shallow_prompts < self.num_prompts

        self.prompt_tokens = nn.Parameter(torch.zeros(depth, self.num_prompts, embed_dim))
        # frozen shallow tokens
        self.last_prompt_tokens = nn.Parameter(torch.zeros(1, self.num_prompts, embed_dim))

        self.vpt_drop = nn.ModuleList([nn.Dropout(p=vpt_dropout) for d in range(depth)])

        ### re-initialize positional embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + self.num_prompts, embed_dim))

        trunc_normal_(self.prompt_tokens, std=.02)
        trunc_normal_(self.last_prompt_tokens, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        with torch.no_grad():
            self.mask_vpt_pos_embed()

        return

    def mask_vpt_pos_embed(self):
        self.pos_embed[:, 1:self.num_prompts + 1, :] = 0.0
        return

    def unfreeze_prompt(self):
        for m in self.parameters():
            m.requires_grad = False
        self.prompt_tokens.requires_grad = True
        self.last_prompt_tokens.requires_grad = True
        return

    def unfreeze_last_prompt(self):
        for m in self.parameters():
            m.requires_grad = False
        self.last_prompt_tokens.requires_grad = True
        return

    def load_from_state_dict(self, state_dict, strict=False):
        """ load state_dict from DINO pre-trained model
        """
        init_weight = self.pos_embed.data
        pos_embed = state_dict.pop('pos_embed')  # manual loading
        init_weight[0, 0, :] = pos_embed[0, 0, :]
        init_weight[0, 1 + self.num_prompts:, :] = pos_embed[0, 1:, :]
        self.pos_embed.data = init_weight
        self.load_state_dict(state_dict, strict=strict)
        return

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1 - self.num_prompts
        N = self.pos_embed.shape[1] - 1 - self.num_prompts
        if npatch == N and w == h:
            return self.pos_embed

        ### TODO: test, corrected
        class_pos_embed = self.pos_embed[:, :1 + self.num_prompts, :]
        patch_pos_embed = self.pos_embed[:, 1 + self.num_prompts:, :]
        # print(f'class_pos_embed={class_pos_embed.shape} patch_pos_embed={patch_pos_embed.shape}')
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # B, L, D

        # add the [CLS] and [PROMPT] token to the embed patch tokens
        cls_token = self.cls_token.expand(B, -1, -1)
        prompt_tokens = self.prompt_tokens[0].expand(B, -1, -1)
        prompt_tokens = self.vpt_drop[0](prompt_tokens)
        x = torch.cat((cls_token, prompt_tokens, x), dim=1)  # B, 1+P+L, D

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)


    def forward(self, x, return_all_patches=False):
        scales = [1, 2]
        output_shape = 'bnc'
        num_prefix_token = 6
        # forward(model, input, scales=None, img_sizes=None, max_split_size=None, resize_output_to_idx=0,
        #         num_prefix_token=0,
        #         output_shape='bnc'):

        assert x.dim() == 4, "Input image must be in the shape of BxCxHxW."
        assert x.shape[2] == x.shape[3], "Currently only square images are supported."
        assert output_shape in ['bnc',
                                'bchw'], "Output shape should be either BxNxC (e.g., ViT) or BxCxHxW (e.g., ConvNet)."
        assert output_shape == 'bnc' or num_prefix_token == 0, "For ConvNet there shouldn't be any prefix token."

        b, c, input_size, _ = input.shape

        # image size for each scale
        assert scales is not None or img_sizes is not None, "Please assign either scales or img_sizes."
        img_sizes = img_sizes or [int(input_size * scale) for scale in scales]

        # prepare multiscale inputs
        max_split_size = max_split_size or input_size  # The maximum size of each split of image. Set as the input size by default
        num_splits = [math.ceil(size / max_split_size) for size in img_sizes]  # number of splits each scale
        input_multiscale = []
        for size, num_split in zip(img_sizes, num_splits):
            x = F.interpolate(input.to(torch.float32), size=size, mode='bicubic').to(input.dtype)
            x = split_chessboard(x, num_split=num_split)
            input_multiscale.append(x)

        # run feedforward on each scale
        outs_multiscale = [model(x) for x in input_multiscale]
        if num_prefix_token > 0:
            outs_prefix_multiscale = [out[:, :num_prefix_token] for out in outs_multiscale]
            outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]
        if output_shape == 'bnc':
            outs_multiscale = [
                rearrange(out, 'b (h w) c -> b c h w', h=int(out.shape[1] ** 0.5), w=int(out.shape[1] ** 0.5))
                for out in outs_multiscale]

        # merge outputs of different splits for each scale separately
        outs_multiscale = [merge_chessboard(out, num_split=num_split) for num_split, out in
                           zip(num_splits, outs_multiscale)]

        # interpolate outputs from different scales and concat together
        output_size = outs_multiscale[resize_output_to_idx].shape[-2]
        out = torch.cat([F.interpolate(outs_multiscale[i].to(torch.float32), size=output_size,
                                       mode='area').to(outs_multiscale[i].dtype)
                         for i in range(len(outs_multiscale))], dim=1)
        if output_shape == 'bnc':
            out = rearrange(out, 'b c h w -> b (h w) c')
        if num_prefix_token > 0:
            # take the mean of prefix tokens from different splits for each scale
            outs_prefix_multiscale = [torch.stack(out.split(b, dim=0), dim=0).mean(dim=0) for out in
                                      outs_prefix_multiscale]
            out_prefix_multiscale = torch.cat(outs_prefix_multiscale, dim=-1)
            out = torch.cat([out_prefix_multiscale, out], dim=1)

        return out



    def single_forward(self, x, return_all_patches=False):

        multiscale_forward

        x = self.prepare_tokens(x)  # B, L, D
        B = x.size(0)
        n_vpt_layer = self.prompt_tokens.size(0) - 1
        for idx_layer, blk in enumerate(self.blocks):
            x = blk(x)
            if idx_layer < n_vpt_layer - 1: # use pretrained frozen prompt
                ### exclude precedent prompts
                a = x[:, 0, :].unsqueeze(1) if self.n_shallow_prompts == 0 else x[:, :1 + self.n_shallow_prompts, :]
                c = x[:, self.num_prompts + 1:, :]
                ### generate prompt input
                b = self.prompt_tokens[idx_layer + 1, self.n_shallow_prompts:, :].expand(B, -1,
                                                                                         -1)  # corrected by i+1, origical i
                b = self.vpt_drop[idx_layer + 1](b)
                x = torch.cat([a, b, c], dim=1)
            elif idx_layer == n_vpt_layer - 1:
                a = x[:, 0, :].unsqueeze(1) if self.n_shallow_prompts == 0 else x[:, :1 + self.n_shallow_prompts, :]
                c = x[:, self.num_prompts + 1:, :]
                ### generate prompt input
                b = self.last_prompt_tokens[0, self.n_shallow_prompts:, :].expand(B, -1,
                                                                                         -1)  # corrected by i+1, origical i
                b = self.vpt_drop[idx_layer + 1](b)
                x = torch.cat([a, b, c], dim=1)

        x = self.norm(x)


        if return_all_patches:
            return x
        else:
            return x[:, 0]

    def get_last_vpt_selfattention(self, x):
        assert self.n_shallow_prompts == 0
        x = self.prepare_tokens(x)
        B = x.size(0)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
                ### exclude precedent prompts
                a = x[:, 0, :].unsqueeze(1)
                c = x[:, self.num_prompts + 1:, :]
                b = self.prompt_tokens[i, :, :].expand(B, -1, -1)
                x = torch.cat([a, b, c], dim=1)
            else:
                # return attention of the last block
                x, attn = blk(x, return_attention=True)
                x = self.norm(x)
                return x, attn

    def get_intermediate_layers(self, x):
        assert self.n_shallow_prompts == 0
        x = self.prepare_tokens(x)
        B = x.size(0)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            output.append(self.norm(x))
            ### exclude precedent prompts
            a = x[:, 0, :].unsqueeze(1)
            c = x[:, self.num_prompts + 1:, :]
            b = self.prompt_tokens[i, :, :].expand(B, -1, -1)
            x = torch.cat([a, b, c], dim=1)
        output = torch.stack(output, dim=1)
        return output
# if __name__=='__main__':
#     model = VPT_ViT(num_prompts=2, vpt_type='shallow')
#     model.unfreeze_prompt()
#     x = torch.rand(16,3,224,224)
#     y = model(x)
#     y.sum().backward()
#     print(f'y={y.shape}')

#     model = VPT_ViT(num_prompts=2, vpt_type='deep')
#     model.unfreeze_prompt()
#     x = torch.rand(16,3,224,224)
#     y = model(x)
#     y.sum().backward()
#     print(f'y={y.shape}')