import logging
import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from open_clip import create_model
from open_clip.model import get_cast_dtype
from training.distributed import is_master
from training.precision import get_autocast
from training.zero_shot import multi_gpu_sync
import os
import torch.nn as nn
import matplotlib.pyplot as plt
def run_seg(model, data, args):
    dataloader=data['val'].dataloader
    model.eval()
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    cls_embeddings = dataloader.dataset.embeddings
    cls_embeddings = F.normalize(torch.from_numpy(cls_embeddings).float(), dim=-1)
    cls_embeddings = cls_embeddings.to(args.device)
    if cast_dtype is not None:
        cls_embeddings = cls_embeddings.to(dtype=cast_dtype)
    unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    baseline_model = create_model(
        args.model,  # same !
        args.pretrained,
        device=args.device,
        precision=args.precision,
        output_dict=True,
        cache_dir="ckpt_path"  # cache dir of pre-trained models
    ).to(args.device)
    with torch.no_grad():
        # _, images, bboxes, image_crops, gt_masks, masked_image_crops, proxy_imgs
        # for images, gt_masks, image_names, image_shapes in tqdm(dataloader, disable=not is_master(args)):
        logging.info('Region classifier')
        for image_names, images, _, _, gt_masks, _, _ in tqdm(data['val'].dataloader, disable=not is_master(args)):
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            tar_h,tar_w=images.shape[-2]//4,images.shape[-1]//4
            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    module = model.module
                    baseline_module=baseline_model.module
                else:
                    module = model
                    baseline_module=baseline_model
                feature_maps = module.encode_dense(images,
                                    normalize=True,
                                    keep_shape=False,
                                    mode="ss",
                                    ex_feats=None)

                baseline_feature_maps = baseline_module.encode_dense(images,
                                                    normalize=True,
                                                    keep_shape=False,
                                                    mode="only_v",
                                                    ex_feats=None)

                bs, N, C = feature_maps.shape
                h, w = int(math.sqrt(N)), int(math.sqrt(N))

                # 计算 logits
                logits = (feature_maps @ cls_embeddings.T) * 40
                baseline_logits = (baseline_feature_maps @ cls_embeddings.T) * 40

                # 变换形状
                logits = logits.permute(0, 2, 1).reshape(bs, -1, h, w)
                baseline_logits = baseline_logits.permute(0, 2, 1).reshape(bs, -1, h, w)

                # 插值
                seg_logits = nn.functional.interpolate(logits, size=(tar_h, tar_w), mode='bilinear')
                baseline_seg_logits = nn.functional.interpolate(baseline_logits, size=(tar_h, tar_w), mode='bilinear')

                batch_size = seg_logits.shape[0]

                for i in range(batch_size):
                    # 处理主模型的预测
                    seg_logits_i = seg_logits[i] * model.logit_scale
                    seg_logits_i = seg_logits_i.softmax(0)  # n_queries * w * h
                    seg_pred = seg_logits_i.argmax(0, keepdim=True)
                    seg_pred = seg_pred.data.cpu().numpy().squeeze(0)
                    
                    # 处理基线模型的预测
                    baseline_seg_logits_i = baseline_seg_logits[i] * model.logit_scale
                    baseline_seg_logits_i = baseline_seg_logits_i.softmax(0)  # n_queries * w * h
                    baseline_seg_pred = baseline_seg_logits_i.argmax(0, keepdim=True)
                    baseline_seg_pred = baseline_seg_pred.data.cpu().numpy().squeeze(0)

                    # 可视化
                    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
                    ax[0].imshow(unnorm(images)[0].permute(1, 2, 0).detach().cpu())
                    ax[0].axis('off')
                    ax[0].set_title("Original Image", fontsize=14)
                    ax[1].imshow(seg_pred, cmap='turbo')
                    ax[1].axis('off')
                    ax[1].set_title("ss distill Result", fontsize=14)
                    ax[2].imshow(baseline_seg_pred, cmap='turbo')
                    ax[2].axis('off')
                    ax[2].set_title("only_v Result", fontsize=14)
                    plt.tight_layout()
                    log_base_path = os.path.join(args.logs, args.name)
                    plt.savefig(f"{log_base_path}/{image_names[i].split('.')[0]}.jpg", bbox_inches='tight')
                    plt.close()

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        if len(image2.shape) == 4:
            # batched
            image2 = image2.permute(1, 0, 2, 3).contiguous()
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2.permute(1, 0, 2, 3).contiguous()

if __name__=="__main__":
    pass