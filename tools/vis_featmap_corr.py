import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torchvision.transforms as T
from PIL import Image
from featup.util import UnNormalize,remove_axes
from featup.plotting import plot_feats, plot_lang_heatmaps
from open_clip.factory import create_model, get_tokenizer
from open_clip.transform import det_image_transform, image_transform
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.segment_anything import sam_model_registry
from torchvision import transforms

device= torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
image_path = "CECloud_code/test_images/animal (25).jpg"
output_path="test_result"
model_name="EVA02-CLIP-B-16" 
# model_name="EVA02-CLIP-L-14-336"
ckpt="checkpoints/EVA02_CLIP_B_psz16_s8B.pt"
token_choosen = 1505

@torch.no_grad()
def vis_feat_corr(vfm="sam"):
    image_size=1024 if model_name=="EVA02-CLIP-B-16" else 896
    vlm = create_model(
           model_name,
            "eva",
            precision="fp32",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=None,
            force_image_size=None,
            pretrained_image=False,
            pretrained_hf=True,
            cache_dir=ckpt,
            output_dict=None).to(device)
    image_mean = getattr(vlm.visual, 'image_mean', None)
    image_std = getattr(vlm.visual, 'image_std', None)
    unnorm = UnNormalize(image_mean, image_std)
    vlm_transform=det_image_transform(
                image_size,
                is_train=False,
                mean=image_mean,
                std=image_std,
                fill_color=0)
    if vfm=="dino":
        resolution = 512
        vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8').half().to(device)
    elif vfm=="dinov2":
        resolution = 896
        vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').half().to(device)
    else:
        resolution = 1024
        vfm = sam_model_registry["vit_b"](checkpoint="").half().to(device)
    vfm_transform=det_image_transform(
            resolution,
            is_train=False,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            fill_color=0)
    raw_img=Image.open(image_path)
    vlm_image = vlm_transform(raw_img).unsqueeze(0).to(device,non_blocking=True)
    vfm_image= vfm_transform(raw_img).unsqueeze(0).to(device,non_blocking=True).half()
    if vfm=="sam":
        vfm_feats=vfm.image_encoder(vfm_image).flatten(2, 3)
    elif vfm=="dino":
        feat = vfm.get_intermediate_layers(vfm_image)[0]
        nb_im = feat.shape[0]
        patch_size = vfm.patch_embed.patch_size
        I, J = vfm_image[0].shape[-2] // patch_size, vfm_image[0].shape[-2] // patch_size
        vfm_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2).flatten(2, 3)
    else: # dinov2
        vfm_feats = vfm.get_intermediate_layers(vfm_image, reshape=True)[0].flatten(2, 3)
    normed_vfm_feats= F.normalize(vfm_feats, dim=1)
    vfm_similarity = torch.einsum("b c m, b c n -> b m n", normed_vfm_feats, normed_vfm_feats)
    vlm_feats, q_feats = vlm.encode_dense(vlm_image,
                            normalize=True,
                            keep_shape=True,
                            mode="ss_vfm_distill",
                            ex_feats=None)
    vfm_feats = vfm_feats.reshape(vlm_feats.shape[0],-1,vlm_feats.shape[2],vlm_feats.shape[3])
    vfm_feats_vis=vfm_feats.squeeze(0).mean(dim=0).detach().cpu().numpy()
    N, _ = q_feats.shape[1:]
    H, W = int(math.sqrt(N)),int(math.sqrt(N))
    q_feats = q_feats.transpose(0, 1).contiguous().view(N, vlm_image.shape[0], -1).transpose(0, 1)
    q_feats = F.normalize(q_feats, dim=-1).transpose(-2,-1)
    vlm_similarity=torch.einsum("b c m, b c n -> b m n", q_feats, q_feats)
    vlm_similarity = vlm_similarity.squeeze(0)[token_choosen, :].view(H,W).cpu().detach().numpy()
    vfm_similarity = vfm_similarity.squeeze(0)[token_choosen, :].view(H,W).cpu().detach().numpy()
    # 计算原图中token的坐标
    token_index = token_choosen  # 选择的token索引
    H, W = 64, 64  # 特征图的大小
    original_size = 1024  # 原图大小
    downsample_factor = original_size // H  # 下采样因子

    # 计算token在特征图中的行列坐标
    row = token_index // H
    col = token_index % H

    # 计算token在原图中的坐标
    original_row = row * downsample_factor
    original_col = col * downsample_factor

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))  
    ax[0].imshow(unnorm(vlm_image)[0].permute(1,2,0).detach().cpu())
    ax[0].axis('off')
    ax[0].set_title('Raw Image')
    # 在原图上绘制标记
    ax[0].scatter(original_col, original_row, color='red', s=100, edgecolor='yellow', marker='o')  # 标记位置


    im1 = ax[1].imshow(vfm_similarity, cmap='jet', interpolation='nearest')
    ax[1].axis('off')
    ax[1].set_title('vfm similarities')
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(vlm_similarity, cmap='jet', interpolation='nearest')
    ax[2].axis('off')
    ax[2].set_title('vlm similarities')
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    ax[3].imshow(vfm_feats_vis, cmap='jet', interpolation='nearest') 
    ax[3].axis('off')
    ax[3].set_title('vfm_feats')

    plt.savefig(os.path.join(output_path,"token_sim.jpg"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__=="__main__":
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    vis_feat_corr(vfm="dinov2")
