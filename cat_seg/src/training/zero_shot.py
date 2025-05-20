import logging
import torch
import torch.nn.functional as F
from training.dist_utils import all_gather
from tqdm import tqdm
from .distributed import is_master
from open_clip import get_cast_dtype
from .precision import get_autocast

def run(model, dataloader, args):
    cls_embeddings = dataloader.dataset.embeddings
    cls_embeddings = F.normalize(torch.from_numpy(cls_embeddings).float(), dim=-1)
    cls_embeddings = cls_embeddings.to(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    if cast_dtype is not None:
        cls_embeddings = cls_embeddings.to(dtype=cast_dtype)
    if 'distill' in args.mode:
        if args.mode=="qq_vfm_distill":
            inference_mode="qq"
        elif args.mode=="kk_vfm_distill":
            inference_mode="kk"
        elif args.mode=="csa_vfm_distill":
            inference_mode="csa"
        elif args.mode=="vv_vfm_distill":
            inference_mode="vv"
        elif args.mode=="all_vfm_distill":
            inference_mode="all"
    elif args.mode=="sanity_check":
        inference_mode="vanilla"
    else:
        inference_mode = args.mode
    with torch.no_grad():
        correct_rois = []
        correct_maskpool = []
        correct_crops = []
        similarity_crops = []
        similarity_rois = []
        similarity_maskpool = []
        all_box_sizes = []
        all_is_thing = []
        all_cls_labels = []
        for _, images, bboxes, image_crops, gt_masks, masked_image_crops in tqdm(dataloader, disable=not is_master(args)):
            images = images.to(args.device)
            bboxes = bboxes.to(args.device)
            image_crops = image_crops.to(args.device)
            masked_image_crops = masked_image_crops.to(args.device)
            gt_masks = gt_masks.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
                bboxes = bboxes.to(dtype=cast_dtype)
                image_crops = image_crops.to(dtype=cast_dtype)
                masked_image_crops = masked_image_crops.to(dtype=cast_dtype)
                gt_masks = gt_masks.to(dtype=cast_dtype)
            image_crops_list = []
            gt_masks_list = []
            cls_labels = []
            rois = []
            box_sizes = []
            is_thing = []
            for bboxes_per_image, crops_per_image, gt_mask, masked_crops_per_image \
                    in zip(bboxes, image_crops, gt_masks, masked_image_crops):
                valid = bboxes_per_image[:, 5] > 0.5
                rois.append(bboxes_per_image[valid, :4])
                cls_labels.append(bboxes_per_image[valid, 4])
                image_crops_list.append(crops_per_image[valid])
                gt_masks_list.append(gt_mask[valid])
                box_sizes.append(bboxes_per_image[valid, 6])
                is_thing.append(bboxes_per_image[valid, 7])
            cls_labels = torch.cat(cls_labels, dim=0).to(torch.long)
            if cls_labels.shape[0] == 0:
                continue
            image_crops = torch.cat(image_crops_list)
            box_sizes = torch.cat(box_sizes, dim=0).float()
            is_thing = torch.cat(is_thing, dim=0)
            all_box_sizes.append(box_sizes)
            all_is_thing.append(is_thing)
            with autocast():
                # predict
                module = model
                roi_extractor = module.encode_pseudo_boxes
                mask_pooler = module.encode_masks
                roi_features = roi_extractor(images, 
                                            rois, 
                                            normalize=True,
                                            mode=inference_mode)
                maskpool_features = mask_pooler(images, 
                                                gt_masks_list,
                                                normalize=True, 
                                                mode=inference_mode)
                # New way to obtain crop features
                if args.image_ave_pool:
                    feature_map = module.visual.encode_dense(image_crops, keep_shape=True)
                    crop_features = feature_map.mean(dim=(-2, -1))
                    crop_features = F.normalize(crop_features, dim=-1)
                else:
                    crop_features = module.encode_image(image_crops, normalize=True)

                if cast_dtype is not None:
                    roi_features = roi_features.to(dtype=cast_dtype)
                    crop_features = crop_features.to(dtype=cast_dtype)
                    maskpool_features = maskpool_features.to(dtype=cast_dtype)
                roi_logits = roi_features @ cls_embeddings.T
                crop_logits = crop_features @ cls_embeddings.T
                maskpool_logits = maskpool_features @ cls_embeddings.T
        
            _, roi_top5_inds = roi_logits.topk(5)
            _, crop_top5_inds = crop_logits.topk(5)
            _, maskpool_top5_inds = maskpool_logits.topk(5)
            correct_rois.append(roi_top5_inds == cls_labels.view(-1, 1))
            correct_crops.append(crop_top5_inds == cls_labels.view(-1, 1))
            correct_maskpool.append(maskpool_top5_inds == cls_labels.view(-1, 1))

            similarity_rois.append(torch.gather(roi_logits, dim=1, index=cls_labels.view(-1, 1))[:, 0])
            similarity_crops.append(torch.gather(crop_logits, dim=1, index=cls_labels.view(-1, 1))[:, 0])
            similarity_maskpool.append(torch.gather(maskpool_logits, dim=1, index=cls_labels.view(-1, 1))[:, 0])
            all_cls_labels.append(cls_labels)
        # TODO: gather correct matrix across gpus
        correct_rois = torch.cat(correct_rois).float()
        correct_crops = torch.cat(correct_crops).float()
        correct_maskpool = torch.cat(correct_maskpool).float()
        similarity_rois = torch.cat(similarity_rois).float()
        similarity_crops = torch.cat(similarity_crops).float()
        similarity_maskpool = torch.cat(similarity_maskpool).float()
        all_box_sizes = torch.cat(all_box_sizes)
        all_is_thing = torch.cat(all_is_thing)
        all_cls_labels = torch.cat(all_cls_labels)
        if args.distributed:
            correct_rois = multi_gpu_sync(correct_rois)
            correct_crops = multi_gpu_sync(correct_crops)
            correct_maskpool = multi_gpu_sync(correct_maskpool)
            all_box_sizes = multi_gpu_sync(all_box_sizes)
            all_is_thing = multi_gpu_sync(all_is_thing)
            similarity_rois = multi_gpu_sync(similarity_rois)
            similarity_crops = multi_gpu_sync(similarity_crops)
            similarity_maskpool = multi_gpu_sync(similarity_maskpool)
            all_cls_labels = multi_gpu_sync(all_cls_labels)

    return correct_rois, correct_crops, correct_maskpool, \
        similarity_rois, similarity_crops, similarity_maskpool, \
        all_box_sizes, all_is_thing, all_cls_labels


def multi_gpu_sync(x):
    device = x.device
    x_list = all_gather(x.cpu())
    x = torch.cat([res.to(device) for res in x_list])
    return x


def macc_with_is_thing(correct_matrix, is_thing, all_cls_labels, prefix):
    def _macc(corrects, cls_labels):
        min_id = cls_labels.min().item()
        max_id = cls_labels.max().item()
        cand_labels = list(range(min_id, max_id+1))

        acc_per_cls = []

        for lb in cand_labels:
            corrects_per_cls = corrects[cls_labels == lb]
            if corrects_per_cls.shape[0] == 0:
                continue
            acc_per_cls.append(corrects_per_cls.mean().half().item())

        return sum(acc_per_cls) / len(acc_per_cls)

    results = {}
    thing_correct_matrix = correct_matrix[is_thing > 0]
    stuff_correct_matrix = correct_matrix[is_thing < 1]

    thing_cls_labels = all_cls_labels[is_thing > 0].long()
    stuff_cls_labels = all_cls_labels[is_thing < 1].long()

    thing_top1_acc = _macc(thing_correct_matrix[:, 0], thing_cls_labels)
    thing_top5_acc = _macc(thing_correct_matrix.sum(-1), thing_cls_labels)

    stuff_top1_acc = _macc(stuff_correct_matrix[:, 0], stuff_cls_labels)
    stuff_top5_acc = _macc(stuff_correct_matrix.sum(-1), stuff_cls_labels)

    results[f'{prefix}.thing.macc1'] = thing_top1_acc
    results[f'{prefix}.thing.macc5'] = thing_top5_acc
    results[f'{prefix}.stuff.macc1'] = stuff_top1_acc
    results[f'{prefix}.stuff.macc5'] = stuff_top5_acc

    return results


def zero_shot_eval(model, data, epoch, args):
    if 'val' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    logging.info('Region classifier')
    results = {}
    correct_rois, correct_crops, correct_maskpool, \
        similarity_rois, similarity_crops, similarity_maskpool, \
        all_box_sizes, all_is_thing, all_cls_labels = run(model, data['val'].dataloader, args)
    results.update(macc_with_is_thing(correct_rois, all_is_thing, all_cls_labels, 'rois'))
    results.update(macc_with_is_thing(correct_crops, all_is_thing, all_cls_labels, 'crops'))
    results.update(macc_with_is_thing(correct_maskpool, all_is_thing, all_cls_labels, 'maskpool'))

    return results
