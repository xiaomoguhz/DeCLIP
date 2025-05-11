
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext
from src.segment_anything import sam_model_registry

def get_autocast(precision):
    if precision == "bf16":
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16) 
    elif precision == "amp":
        return lambda: torch.cuda.amp.autocast() 
    else:
        return lambda: nullcontext() 
    
def mask2box(mask):
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return x0, y0, x1, y1

def freeze_parameters(model,args):
    freeze_keys = get_freeze_keys(args)
    for name, param in model.named_parameters():
        if name in freeze_keys:
            param.requires_grad = False
    return model

def build_vfm(name):
    sam_ckpts = {
        "sam-B": "sam_vit_b_01ec64.pth",
        "sam-L": "sam_vit_l_0b3195.pth", 
    }
    
    dinov2_ckpts = {
        "dinov2-L": "dinov2_vitl14_reg",
        "dinov2-B": "dinov2_vitb14_reg",
    }

    dino_ckpts = {
        "dino-B-8": "dino_vitb8",
        "dino-B-16": "dino_vitb16",
    }

    vfm = None

    # SAM
    if name.startswith("sam"):
        if name in sam_ckpts:
            vit_type = "vit_b" if "B" in name else "vit_l"
            checkpoint_name = sam_ckpts[name]
            try:
                vfm = sam_model_registry[vit_type](checkpoint=checkpoint_name).half()
            except Exception as e:
                raise RuntimeError(f"Failed to load SAM model '{name}' with checkpoint '{checkpoint_name}': {e}")
        else:
            raise NotImplementedError(f"VLM model '{name}' not supported under SAM category.")

    # DINOv2
    elif name.startswith("dinov2"):
        if name in dinov2_ckpts:
            model_name = dinov2_ckpts[name]
            try:
                vfm = torch.hub.load(
                    'facebookresearch/dinov2',
                    model_name,
                    source='github'  
                ).half()
            except Exception as e:
                raise RuntimeError(f"Failed to load DINOv2 model '{name}': {e}")
        else:
            raise NotImplementedError(f"VLM model '{name}' not supported under DINOv2 category.")

    # DINO
    elif name.startswith("dino"):
        if name in dino_ckpts:
            model_name = dino_ckpts[name]
            try:
                vfm = torch.hub.load(
                    'facebookresearch/dino',
                    model_name,
                    source='github'
                ).half()
            except Exception as e:
                raise RuntimeError(f"Failed to load DINO model '{name}': {e}")
        else:
            raise NotImplementedError(f"VLM model '{name}' not supported under DINO category.")

    else:
        raise NotImplementedError(f"VLM model '{name}' not supported.")

    for p in vfm.parameters():
        p.requires_grad = False

    return vfm


def get_freeze_keys(args):
    if args.model=="ViT-B-16":
        return ViTB_16_freeze_keys
    elif args.model=="ViT-L-14" or args.model=="ViT-L-14-336":
         return ViTL_14_freeze_keys
    elif args.model=="EVA02-CLIP-B-16":
        if args.mode=="qq_vfm_distill":
            return ViTB_EVA_16_qq_Distill_keys
        elif args.mode=="kk_vfm_distill":
            return ViTB_EVA_16_kk_Distill_keys
        elif args.mode=="sanity_check":
            return sanity_check_freeze_keys
        else: 
            return BASE_EVA_ViTB_16_freeze_keys
    elif args.model=="EVA02-CLIP-L-14-336":
        if args.mode=="qq_vfm_distill":
            return ViTL_EVA_14_qq_Distill_keys
        elif args.mode=="kk_vfm_distill":
            return ViTL_EVA_14_kk_Distill_keys
        elif args.mode=="sanity_check":
            return sanity_check_freeze_keys
        else: 
            return BASE_EVA_ViTL_14_freeze_keys
    elif args.model=="siglip-so400m-patch14-384":
        return siglip_384_Distill_Freeze_keys

ViTB_16_freeze_keys=[
             'visual.transformer.resblocks.11.ln_2.weight',
             'visual.transformer.resblocks.11.ln_2.bias',
             'visual.transformer.resblocks.11.mlp.c_fc.weight',
             'visual.transformer.resblocks.11.mlp.c_fc.bias',
             'visual.transformer.resblocks.11.mlp.c_proj.weight',
             'visual.transformer.resblocks.11.mlp.c_proj.bias']

ViTL_14_freeze_keys=[
             'visual.transformer.resblocks.23.ln_2.weight',
             'visual.transformer.resblocks.23.ln_2.bias',
             'visual.transformer.resblocks.23.mlp.c_fc.weight',
             'visual.transformer.resblocks.23.mlp.c_fc.bias',
             'visual.transformer.resblocks.23.mlp.c_proj.weight',
             'visual.transformer.resblocks.23.mlp.c_proj.bias']

BASE_EVA_ViTB_16_freeze_keys=[
            'logit_scale',
             'visual.blocks.11.norm2.weight',
             'visual.blocks.11.norm2.bias',
             'visual.blocks.11.mlp.w1.weight',
             'visual.blocks.11.mlp.w1.bias',
             'visual.blocks.11.mlp.w2.weight',
             'visual.blocks.11.mlp.w2.bias',
             'visual.blocks.11.mlp.w3.weight',
             'visual.blocks.11.mlp.w3.bias',
             'visual.blocks.11.mlp.ffn_ln.weight',
             'visual.blocks.11.mlp.ffn_ln.bias']

BASE_EVA_ViTL_14_freeze_keys=[
             'logit_scale',
             'visual.blocks.23.norm2.weight',
             'visual.blocks.23.norm2.bias',
             'visual.blocks.23.mlp.w1.weight',
             'visual.blocks.23.mlp.w1.bias',
             'visual.blocks.23.mlp.w2.weight',
             'visual.blocks.23.mlp.w2.bias',
             'visual.blocks.23.mlp.w3.weight',
             'visual.blocks.23.mlp.w3.bias',
             'visual.blocks.23.mlp.ffn_ln.weight',
             'visual.blocks.23.mlp.ffn_ln.bias']


sanity_check_freeze_keys=['logit_scale']
ViTB_EVA_16_qq_Distill_keys=['visual.blocks.11.attn.k_proj.weight',
                             ] + BASE_EVA_ViTB_16_freeze_keys
ViTL_EVA_14_qq_Distill_keys=['visual.blocks.23.attn.k_proj.weight',
                             ] + BASE_EVA_ViTL_14_freeze_keys
ViTB_EVA_16_kk_Distill_keys=['visual.blocks.11.attn.q_proj.weight','visual.blocks.11.attn.q_bias'] + BASE_EVA_ViTB_16_freeze_keys
ViTL_EVA_14_kk_Distill_keys=['visual.blocks.23.attn.q_proj.weight','visual.blocks.23.attn.q_bias'] + BASE_EVA_ViTL_14_freeze_keys

siglip_384_Distill_Freeze_keys=['logit_scale',
                                'logit_bias',
                                'vision_model.head.probe',
                                # 'vision_model.head.attention.out_proj.weight',
                                # 'vision_model.head.attention.out_proj.bias',
                                # 'vision_model.head.layernorm.weight',
                                # 'vision_model.head.layernorm.bias',
                                # 'vision_model.head.mlp.fc1.weight',
                                # 'vision_model.head.mlp.fc1.bias',
                                # 'vision_model.head.mlp.fc2.weight',
                                # 'vision_model.head.mlp.fc2.bias',
                                ]