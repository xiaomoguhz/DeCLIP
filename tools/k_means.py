import logging
import cv2
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from open_clip.model import get_cast_dtype
from training.distributed import is_master
from training.precision import get_autocast
from training.zero_shot import multi_gpu_sync
import os

def run_kmeans(model, data, args):
    model.eval()
    def _process_cluster(cluster, h, w):
        cluster = cluster.reshape(h, w).astype(np.float32)
        cluster = cv2.medianBlur(cluster, 5)
        return cluster.reshape(h*w) > 0.5

    def _per_image_kmeans(feature_map, masks, image_name, image_shape):
        f_h, f_w = feature_map.shape[1:] # 64, 64
        tar_h,tar_w = tuple(image_shape.tolist())# 1024, 1024
        tar_h,tar_w=tar_h//4,tar_w//4
        # scale_factor = min(f_h/ori_h, f_w/ori_w) # 0.0625
        # tar_h, tar_w = min(int(ori_h * scale_factor), f_h), min(int(ori_w * scale_factor), f_w)
        # feature_map = feature_map[:, :tar_h, :tar_w].contiguous().view(-1, tar_h * tar_w).T
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(tar_h, tar_w),mode="bilinear").squeeze(0).contiguous().view(-1, tar_h * tar_w).T
        valid = masks.sum((-2, -1)) > 0
        masks = masks[valid, :tar_h, :tar_w]
        if masks.shape[0] == 0:
            return torch.tensor([]).to(feature_map)
        masks = masks.view(-1, tar_h * tar_w).to(feature_map)
        # TODO: kmeans on feature_map
        feature_map = F.normalize(feature_map, dim=-1).cpu().numpy()
        cluster_method = KMeans(n_clusters=len(masks), n_init=10)
        # fit model and predict clusters
        results = cluster_method.fit_predict(feature_map)
        cluster_ids = np.unique(results)
        clusters = np.stack([_process_cluster(results == cluster_id, tar_h, tar_w) for cluster_id in cluster_ids if cluster_id >= 0])
        clusters = torch.from_numpy(clusters).to(masks)
        union = torch.clamp(clusters[:, None] + masks[None], max=1.0).sum(-1)
        intersection = (clusters[:, None] * masks[None]).sum(-1)
        iofs = intersection / (union + 1e-12)
        max_iofs = iofs.max(dim=-1).values
        # TODO: save the results
        results = results.reshape(tar_h, tar_w)
        log_base_path = os.path.join(args.logs, args.name)
        np.save(f"{log_base_path}/{image_name.split('.')[0]}.npy", results)
        return max_iofs
    
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with torch.no_grad():
        best_overlaps = []
        # _, images, bboxes, image_crops, gt_masks, masked_image_crops, proxy_imgs
        # for images, gt_masks, image_names, image_shapes in tqdm(dataloader, disable=not is_master(args)):
        logging.info('Region classifier')
        for image_names, images, _, _, gt_masks, _, _ in tqdm(data['val'].dataloader, disable=not is_master(args)):
            image_shapes=[]
            for image in images:
                image_shapes.append(torch.tensor([image.shape[-2],image.shape[-1]],device=images.device))
            image_shapes=torch.stack(image_shapes,dim=0)
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    module = model.module
                else:
                    module = model
                feature_maps = module.encode_dense(images,
                                                    normalize=True,
                                                    keep_shape=True,
                                                    mode="ss",
                                                    ex_feats=None)
            best_overlaps += list(map(_per_image_kmeans, feature_maps, gt_masks, image_names, image_shapes))
        best_overlaps = torch.cat(best_overlaps)
        if args.distributed and not args.horovod:
            best_overlaps = multi_gpu_sync(best_overlaps)

    return best_overlaps.mean()



if __name__=="__main__":
    pass