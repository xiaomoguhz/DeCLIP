from transformers import SiglipModel,SiglipVisionModel
from transformers.models.siglip.modeling_siglip import (SiglipVisionTransformer,SiglipMultiheadAttentionPoolingHead,SiglipVisionEmbeddings)
from typing import Optional, Tuple, Union
from transformers.models.siglip.configuration_siglip import SiglipConfig, SiglipVisionConfig
import torch.nn as nn
import torch 
from transformers.modeling_outputs import BaseModelOutput,BaseModelOutputWithPooling
import torch.nn.functional as F
import math
from torchvision.ops import roi_align
from transformers import SiglipImageProcessor
from transformers.utils import logging,torch_int
logger=logging.get_logger()
import math 

class DeCLIP_SiglipModel(SiglipModel):
    """
    A modified version of SiglipModel to support dense feature extraction and decoupled distillation.
    Author:
        Junjie Wang
    """
    def __init__(self, config: SiglipConfig):
        super().__init__(config)
        vision_config = config.vision_config
        vision_model = DeCLIP_SiglipVisionModel._from_config(vision_config)
        self.vision_model = vision_model.vision_model
        self.post_init()
        for param in self.text_model.parameters():
            param.requires_grad = False
        logger.info("frozen all the text model")

    def encode_image(self,pixel_values, normalize=False):
        pooled_output= self.get_image_features(pixel_values,None,None,None, interpolate_pos_encoding = False)
        return F.normalize(pooled_output, dim=-1) if normalize else pooled_output
    
    def encode_dense(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        keep_shape: bool = False,
        normalize: bool = True,
    ) -> torch.FloatTensor:
        dense_feats = self.vision_model.encode_dense(pixel_values=pixel_values)
        if normalize:
            dense_feats = F.normalize(dense_feats, dim=-1) # bs,L,h_s
        if keep_shape:
            h = w = int(math.sqrt(dense_feats.shape[1])) # h,w
            dense_feats = dense_feats.view(dense_feats.shape[0], h, w, -1).permute(0, 3, 1, 2)
        return dense_feats

    def encode_pseudo_boxes(self, pixel_values, normed_boxes, normalize, mode):
        if "distill" in mode:
            box_features, context = self.vision_model.extract_roi_features(pixel_values, normed_boxes, mode)
            if normalize:
                box_features = F.normalize(box_features, dim=-1)
            return box_features, context
        else:
            box_features = self.vision_model.extract_roi_features(pixel_values, normed_boxes, mode)
            if normalize:
                box_features = F.normalize(box_features, dim=-1)
            return box_features
    
    def encode_masks(self, image, masks, normalize,mode):
        mask_pooled = self.vision_model.mask_pool(image, masks)
        if normalize:
            mask_pooled = F.normalize(mask_pooled, dim=-1)
        return mask_pooled
    
    def lock_image_tower(self, unlocked_groups=0,freeze_bn_stats=False):
        self.vision_model.lock(unlocked_groups=unlocked_groups)

class DeCLIP_SiglipVisionModel(SiglipVisionModel):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        self.vision_model = DeCLIP_SiglipVisionTransformer(config)
        self.post_init()

class DeCLIP_SiglipVisionTransformer(SiglipVisionTransformer):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        self.embeddings = DeCLIP_SiglipVisionEmbeddings(config)
        if self.use_head:
            self.head = DeCLIPMultiheadAttentionPoolingHead(config)
  
    def lock(self, unlocked_groups=0):
        for param in self.parameters():
            param.requires_grad = False
        def _unlock(x):
            if isinstance(x, list):
                for g in x:
                    _unlock(g)
            else:
                if isinstance(x, torch.nn.Parameter):
                    x.requires_grad = True
                else:
                    for p in x.parameters():
                        p.requires_grad = True

        for blk in self.encoder.layers[-unlocked_groups:]:
            _unlock(blk)
        # default unlock head. part frozen by latter DeCLIP method
        _unlock(self.post_layernorm)
        _unlock(self.head)

    def extract_roi_features(self, x, normed_boxes,mode):
        dense_feats,context = self.encode_dense(x) # bs, L, hs
        bs, L, hs=dense_feats.shape
        h=w=int(math.sqrt(L))
        dense_feats=dense_feats.transpose(-2,-1).view(bs,hs,h,w)
        box_feats=roi_align(dense_feats, self._denormalize_boxes(normed_boxes, dense_feats), (1, 1),1.0, -1, True)[..., 0, 0]
        if "distill" in mode:
            return box_feats, context
        else:
            return box_feats
    
    def mask_pool(self, x, masks):
        dense_feats,_ = self.encode_dense(x)
        num_masks_per_image = [len(masks_per_image) for masks_per_image in masks]
        masks = torch.cat(masks).float().flatten(-2, -1)    # bs, h*w
        dense_feats = torch.repeat_interleave(dense_feats, torch.tensor(num_masks_per_image, device=dense_feats.device), dim=0)
        return (dense_feats * masks.unsqueeze(-1)).sum(1) / (masks.sum(1, keepdim=True) + 1e-12)
    
    def encode_dense(self, pixel_values,) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        function that call self-self attn at the last layer
        """
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=True) 
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        last_hidden_state,context = self.head.dense_proj(last_hidden_state) if self.use_head else None
        return (last_hidden_state, context)
    
    @staticmethod
    def _denormalize_boxes(normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        for boxes in normed_boxes:
            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes)
        return denormed_boxes
    
class DeCLIPMultiheadAttentionPoolingHead(SiglipMultiheadAttentionPoolingHead):

    def dense_proj(self, hidden_state):
        attn_layer = self.attention
        num_heads = attn_layer.num_heads
        hidden_state=hidden_state.transpose(0,1)
        _, bsz, embed_dim = hidden_state.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5
        q, k, v = F.linear(hidden_state, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        context = q
        attn_context = torch.bmm(context, context.transpose(1, 2)) *  scale
        attn_weights = F.softmax(attn_context, dim=-1)
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output).transpose(0,1)
        hidden_state = self.layernorm(attn_output)
        hidden_state = hidden_state + self.mlp(hidden_state)
        
        return hidden_state, context

class DeCLIP_SiglipVisionEmbeddings(SiglipVisionEmbeddings):

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        orig_dtype = patch_pos_embed.dtype
        if orig_dtype == torch.bfloat16:
            patch_pos_embed = patch_pos_embed.to(torch.float32)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        if orig_dtype == torch.bfloat16:
            patch_pos_embed = patch_pos_embed.to(torch.bfloat16)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

def create_model(model_name,
                  precision,
                  device,):
    model_name = "google/"+ model_name

    if precision == "bf16":
        torch_dtype = torch.bfloat16  
    elif precision == "amp":
        torch_dtype = torch.float16  
    else:
        torch_dtype = torch.float32
        
    model = DeCLIP_SiglipModel.from_pretrained(model_name,
                                                torch_dtype=torch_dtype,
                                                device_map=device)
    siglip_image_processor = SiglipImageProcessor.from_pretrained(model_name)
    model.vision_model.image_mean = siglip_image_processor.image_mean
    model.vision_model.image_std = siglip_image_processor.image_std
    size = siglip_image_processor.size
    image_size=(size['height'],size['width'])
    model.vision_model.image_size=image_size
    return model