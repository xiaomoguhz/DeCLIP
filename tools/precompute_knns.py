import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from training.distributed import is_master
from training.precision import get_autocast
import os
import json
@torch.no_grad()
def run_knns(model, data, args):
    """
    基于 vfm 的相似度, 计算每个训练集样本的 KNN 样本, 并建立 image_id -> KNN 映射
    """
    model.eval()
    autocast = get_autocast(args.precision)
    knn_mapping_file = os.path.join("", "knn_{}.json".format(args.use_vfm))  # KNN 映射文件
    all_feats = []
    all_image_ids = []

    with torch.no_grad():
        for _, _, _, proxy_image, image_ids in tqdm(data['train'].dataloader, disable=not is_master(args)):
            proxy_image = proxy_image.to(args.device)
            with autocast():
                # predict
                if args.use_vfm == "sam":
                    vfm_feats = model.image_encoder(proxy_image).flatten(2, 3)
                elif args.use_vfm == "dino":
                    feat = model.get_intermediate_layers(proxy_image)[0]
                    nb_im = feat.shape[0]
                    patch_size = model.patch_embed.patch_size
                    I, J = proxy_image[0].shape[-2] // patch_size, proxy_image[0].shape[-2] // patch_size
                    vfm_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2).flatten(2, 3)
                else:  # dinov2
                    vfm_feats = model.get_intermediate_layers(proxy_image, reshape=True)[0].flatten(2, 3)

            vfm_feats = vfm_feats.mean(dim=-1)
            vfm_feats = F.normalize(vfm_feats, dim=1)

            all_feats.append(vfm_feats.to("cpu", non_blocking=True))
            all_image_ids.extend(image_ids.cpu().tolist())  # 将 image_ids 添加到列表中

    normed_feats = torch.cat(all_feats, dim=0).contiguous()

    all_nns = []
    step = normed_feats.shape[0] // args.batch_size

    for i in tqdm(range(0, normed_feats.shape[0], step)):
        batch_feats = normed_feats[i:i + step, :]
        pairwise_sims = torch.einsum("nf,mf->nm", batch_feats, normed_feats)  # 计算余弦相似度
        all_nns.append(torch.topk(pairwise_sims, 30)[1])  # 取前 30 个最近邻的索引
        del pairwise_sims

    nearest_neighbors = torch.cat(all_nns, dim=0)

    knn_dict = {}
    for i, image_id in enumerate(all_image_ids):
        knn_indices = nearest_neighbors[i]
        knn_image_ids = [all_image_ids[idx] for idx in knn_indices]
        knn_dict[image_id] = knn_image_ids

    with open(knn_mapping_file, 'w') as f:
        json.dump(knn_dict, f)
    
    print(f"KNN mapping saved to {knn_mapping_file}")




if __name__=="__main__":
    pass