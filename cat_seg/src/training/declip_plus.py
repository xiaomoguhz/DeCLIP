from typing import List
import torch
import torch.nn.functional as F
import torch
import random

from third_party.utils.utils_correspondence import pca


class DeCLIP_PLUS:
    def __call__(self, batch, student, teacher, vfm_model, args):
        """
        DeCLIP_PLUS 的核心调用函数，计算损失并返回。
        """
        if args.use_vfm=='sd_dino':
            sd_model=vfm_model[1]
            vfm_model=vfm_model[0]
        losses = {}
        context_weight = args.loss_context_weight
        content_weight = args.loss_content_weight   
        cross_context_weight = 0.1

        if args.distributed:
            student = student.module

        dtype_map = {"bf16": torch.bfloat16, "amp": torch.float16}
        input_dtype = dtype_map.get(args.precision, torch.float32)

        images, normed_boxes, image_crops, vfm_image, sd_image, knn_image_vfm, knn_image_clip = prepare_inputs(batch, args.device, input_dtype)
        loss_context, loss_content = intra_image_distill(images, normed_boxes, image_crops, vfm_image,student,teacher, vfm_model,sd_model,sd_image,args,return_context=False)
        # cross_loss_context = cross_image_distill(context, vfm_feats, knn_image_vfm, knn_image_clip, student, vfm_model, args)
        
        losses.update({"loss_context":loss_context * context_weight})
        losses.update({"loss_content":loss_content * content_weight})
        # losses.update({"loss_context_cross": cross_loss_context * cross_context_weight})

        return losses, len(images)

def intra_image_distill(images, normed_boxes, image_crops, proxy_image,student,teacher, vfm_model, sd_model, sd_image, args, return_context=False):
    B = images.shape[0]
    rois_list = []
    crops_list = []
    for bboxes_per_image, crops_per_image in zip(normed_boxes, image_crops):
        valid = bboxes_per_image[:, -1] > 0.5
        rois_list.append(bboxes_per_image[valid, :4])
        crops_list.append(crops_per_image[valid])
    image_crops = torch.cat(crops_list)

    student_roi_features, context = student.encode_pseudo_boxes(images, rois_list, normalize=True, mode = args.mode)
    with torch.no_grad():
        teacher_crop_features = teacher.encode_image(image_crops, normalize=True)
        vfm_feats = extract_vfm_features(vfm_model,proxy_image,args)  # bs, dino_channel, h,w
        bs, c1,h,w= vfm_feats.shape
        vfm_feats = vfm_feats.view(bs, c1, -1)
        vfm_feats = F.normalize(vfm_feats, dim=1) 
        vfm_self_corr = torch.einsum('bci,bcj->bij', vfm_feats, vfm_feats)
        sd_feats=sd_model(sd_image,raw=True)
        with torch.amp.autocast(vfm_feats.device.type, enabled=False):
            sd_feats=pca(sd_feats)
        sd_feats = F.interpolate(sd_feats, size = (h, w), mode='bilinear', align_corners=False) # bs, sd_channel, h,w
        bs, c2 = sd_feats.shape[:2]
        sd_feats = sd_feats.view(bs, c2, -1)
        sd_feats = F.normalize(sd_feats, dim=1)
        sd_self_corr = torch.einsum('bci,bcj->bij', sd_feats, sd_feats)

    student_context_correlations = compute_student_intra_image_similarity(images.shape[0], context, args)

    loss_context_sd=(student_context_correlations - sd_self_corr).norm(p=2,dim=-1).mean()*0.1
    loss_context_vfm=(student_context_correlations - vfm_self_corr).norm(p=2,dim=-1).mean()*0.9
    loss_context=loss_context_sd + loss_context_vfm

    loss_content = 1.0 - (student_roi_features * teacher_crop_features).sum(-1).mean()
    if return_context:
        return loss_context, loss_content, context, vfm_feats
    else:
        return loss_context, loss_content

def cross_image_distill(student_context_orig, vfm_feats_orig, knn_image_vfm, knn_image_clip, student, vfm_model, args):
    b_knn_value=0.2
    b_neg_value=0.5
    B=knn_image_vfm.shape[0]
    N=student_context_orig[0].shape[1] if isinstance(student_context_orig, tuple) else student_context_orig.shape[1]
    # knn part 
    student_context_knn =student.encode_dense(knn_image_clip,normalize=True,keep_shape=True, mode=args.mode)[1]
    student_cross_correlations= compute_student_cross_image_similarity(B, student_context_orig, student_context_knn, args)
    student_cross_correlations_clamped = torch.clamp(student_cross_correlations, min=0.0)
    with torch.no_grad():
        
        vfm_feats_knn  = extract_vfm_features(vfm_model,knn_image_vfm,args)
        vfm_feats_knn = vfm_feats_knn.flatten(start_dim=-2)
        vfm_feats_knn = F.normalize(vfm_feats_knn , dim=1)
        vfm_cross_correlations = torch.einsum("b d m, b d n -> b m n", vfm_feats_orig, vfm_feats_knn)
        vfm_cross_correlations_centered = vfm_cross_correlations - vfm_cross_correlations.mean(dim=2, keepdim=True)

    # neg part 
    # 确保每张图像从 batch 中随机采样一张负样本，但不采样到自己
    student_context_orig=[i.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1) for i in student_context_orig]
    random_indices = []
    for i in range(B):
        valid_indices = list(range(B))
        valid_indices.remove(i) 
        random_indices.append(random.choice(valid_indices)) 
    vfm_random_neg_feats=vfm_feats_orig[random_indices]

    student_random_neg_feats=[i[random_indices] for i in student_context_orig]
    neg_student_cross_correlations=compute_student_cross_image_similarity(B, student_context_orig, student_random_neg_feats, args)
    neg_student_cross_correlations_clamped = torch.clamp(neg_student_cross_correlations, min=0.0)
    with torch.no_grad():
        neg_vfm_cross_correlations = torch.einsum("b d m, b d n -> b m n", vfm_feats_orig, vfm_random_neg_feats)
    neg_vfm_cross_correlations_centered=neg_vfm_cross_correlations-neg_vfm_cross_correlations.mean(dim=2, keepdim=True)

    loss_context_knn = -torch.mean((vfm_cross_correlations_centered - b_knn_value) * student_cross_correlations_clamped)*0.0
    loss_context_neg = -torch.mean((neg_vfm_cross_correlations_centered - b_neg_value) * neg_student_cross_correlations_clamped)
    return loss_context_knn + loss_context_neg

def cross_image_distillv2(student_context_orig, vfm_feats_orig, knn_image_vfm, knn_image_clip, student, vfm_model, args):
    B,N = student_context_orig.shape[:2]
    # knn part 
    student_context_knn =student.encode_dense(knn_image_clip,normalize=True,keep_shape=True, mode=args.mode)[1]
    student_context_knn  = student_context_knn .transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
    student_context_knn  = F.normalize(student_context_knn , dim=-1).transpose(-2, -1)
    with torch.no_grad():
        vfm_feats_knn  = extract_vfm_features(vfm_model,knn_image_vfm,args)
        vfm_feats_knn =vfm_feats_knn.flatten(start_dim=-2)
        vfm_feats_knn = F.normalize(vfm_feats_knn , dim=1)
        vfm_cross_correlations = torch.einsum("b d m, b d n -> b m n", vfm_feats_orig, vfm_feats_knn)
        # 空间中心化: 对每个原始图像token，其与所有KNN图像token的相关性减去均值
        vfm_cross_correlations_centered = vfm_cross_correlations - vfm_cross_correlations.mean(dim=2, keepdim=True)

    student_cross_correlations = torch.einsum("b d m, b d n -> b m n", student_context_orig, student_context_knn.transpose(-2, -1))
    student_cross_correlations_centered= student_cross_correlations - student_cross_correlations.mean(dim=2, keepdim=True)
    # neg part 
    random_indices = []
    for i in range(B):
        valid_indices = list(range(B))
        valid_indices.remove(i)  # 排除自身索引
        random_indices.append(random.choice(valid_indices))  # 从剩余索引中随机采样
    vfm_random_neg_feats=vfm_feats_orig[random_indices]
    student_random_neg_feats=student_context_orig[random_indices]
    with torch.no_grad():
        neg_vfm_cross_correlations = torch.einsum("b d m, b d n -> b m n", vfm_feats_orig, vfm_random_neg_feats)
    neg_student_cross_correlations = torch.einsum("b d m, b d n -> b m n", student_context_orig, student_random_neg_feats)
    neg_student_cross_correlations_centered=neg_student_cross_correlations-neg_student_cross_correlations.mean(dim=2, keepdim=True)
    neg_vfm_cross_correlations_centered=neg_vfm_cross_correlations-neg_vfm_cross_correlations.mean(dim=2, keepdim=True)
    
    loss_context_neg=(neg_vfm_cross_correlations_centered - neg_student_cross_correlations_centered).norm(p=2,dim=-1).mean()
    loss_context_knn=(vfm_cross_correlations_centered - student_cross_correlations_centered).norm(p=2,dim=-1).mean() 
    return (loss_context_knn+loss_context_neg)/2.0

def prepare_inputs(batch, device, dtype):
    """
    将输入批次中的数据加载到设备，并转换为指定数据类型。
    """
    images, normed_boxes, image_crops, proxy_image, sd_image, knn_image_vfm, knn_image_clip = batch
    images = images.to(device=device, dtype=dtype, non_blocking=True)
    normed_boxes = normed_boxes.to(device=device, dtype=dtype, non_blocking=True)
    image_crops = image_crops.to(device=device, dtype=dtype, non_blocking=True)
    proxy_image = proxy_image.to(device=device, dtype=dtype, non_blocking=True)
    knn_image_vfm = knn_image_vfm.to(device=device, dtype=dtype, non_blocking=True)
    knn_image_clip = knn_image_clip.to(device=device, dtype=dtype, non_blocking=True)
    sd_image = sd_image.to(device=device, dtype=dtype, non_blocking=True)
    return images, normed_boxes, image_crops, proxy_image, sd_image, knn_image_vfm, knn_image_clip


def extract_vfm_features(vfm_model, image, args):
    """
    从 VFM 模型中提取特征，并对其进行归一化。
    """
    if 'sam' in args.use_vfm:
        vfm_feats = vfm_model.image_encoder(image)
    elif "dinov2" or "sd_dino" in args.use_vfm:
        vfm_feats = vfm_model.get_intermediate_layers(image, reshape=True)[0]
    elif 'dino' in args.use_vfm:
        feat = vfm_model.get_intermediate_layers(image)[0]
        nb_im = feat.shape[0]
        patch_size = vfm_model.patch_embed.patch_size
        I, J = image[0].shape[-2] // patch_size, image[0].shape[-2] // patch_size
        vfm_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
    else:
        raise NotImplementedError(f"VFM mode {args.use_vfm} is not implemented.")
    return vfm_feats


def compute_student_intra_image_similarity(B, context, args):
    N, _ = context[0].shape[1:] if isinstance(context, tuple) else context.shape[1:]

    if args.mode in ["qq_vfm_distill", "kk_vfm_distill", "vv_vfm_distill", "sanity_check"]:
        context = context.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
        context = F.normalize(context, dim=-1).transpose(-2, -1)
        student_context_similarity = torch.einsum("b c m, b c n -> b m n", context, context)

    elif args.mode == "csa_vfm_distill":
        q_feature, k_feature = context
        q_feature = q_feature.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
        k_feature = k_feature.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
        q_feature = F.normalize(q_feature, dim=-1).transpose(-2, -1)
        k_feature = F.normalize(k_feature, dim=-1).transpose(-2, -1)
        student_context_similarity = (torch.einsum("b c m, b c n -> b m n", q_feature, q_feature) +
                                       torch.einsum("b c m, b c n -> b m n", k_feature, k_feature)) / 2.0
    elif args.mode == "all_vfm_distill":
        q_feature, k_feature, v_feature = context
        features = [q_feature, k_feature, v_feature]
        similarities = []
        for feature in features:
            feature = feature.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
            feature = F.normalize(feature, dim=-1).transpose(-2, -1)
            similarities.append(torch.einsum("b c m, b c n -> b m n", feature, feature))
        student_context_similarity = sum(similarities) / len(features)

    else:
        raise NotImplementedError(f"Mode '{args.mode}' is not implemented.")

    return student_context_similarity


def compute_student_cross_image_similarity(B, img1_context, img2_context, args):
    N, _B = img1_context[0].shape[1:] if isinstance(img1_context, tuple) or isinstance(img1_context, List) else img1_context.shape[1:]

    if args.mode in ["qq_vfm_distill", "kk_vfm_distill", "vv_vfm_distill", "sanity_check"]:
        if not img1_context.shape[0]==B:
            # have not transform
            img1_context = img1_context.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
            img2_context = img2_context.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
        img1_context = F.normalize(img1_context, dim=-1).transpose(-2, -1)
        img2_context = F.normalize(img2_context, dim=-1).transpose(-2, -1)
        student_context_similarity = torch.einsum("b c m, b c n -> b m n", img1_context, img2_context)

    elif args.mode == "csa_vfm_distill":
        img1_q_feature, img1_k_feature = img1_context
        img2_q_feature, img2_k_feature = img2_context
        if not _B == B :
            img1_q_feature = img1_q_feature.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
            img1_k_feature = img1_k_feature.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
            img2_q_feature = img2_q_feature.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
            img2_k_feature = img2_k_feature.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)

        img1_q_feature = F.normalize(img1_q_feature, dim=-1).transpose(-2, -1)
        img1_k_feature = F.normalize(img1_k_feature, dim=-1).transpose(-2, -1)
        img2_q_feature = F.normalize(img2_q_feature, dim=-1).transpose(-2, -1)
        img2_k_feature = F.normalize(img2_k_feature, dim=-1).transpose(-2, -1)

        student_context_similarity = (torch.einsum("b c m, b c n -> b m n", img1_q_feature, img2_q_feature) +
                                       torch.einsum("b c m, b c n -> b m n", img1_k_feature, img2_k_feature)) / 2.0
    elif args.mode == "all_vfm_distill":
        img1_q_feature, img1_k_feature, img1_v_feature = img1_context
        img2_q_feature, img2_k_feature, img2_v_feature = img2_context

        features1 = [img1_q_feature, img1_k_feature, img1_v_feature]
        features2 = [img2_q_feature, img2_k_feature, img2_v_feature]

        similarities = []
        for f1, f2 in zip(features1, features2):
            if not _B==B:
                f1 = f1.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
                f2 = f2.transpose(0, 1).contiguous().view(N, B, -1).transpose(0, 1)
            f1 = F.normalize(f1, dim=-1).transpose(-2, -1)
            f2 = F.normalize(f2, dim=-1).transpose(-2, -1)
            similarities.append(torch.einsum("b c m, b c n -> b m n", f1, f2))

        student_context_similarity = sum(similarities) / len(features1)

    else:
        raise NotImplementedError(f"Mode '{args.mode}' is not implemented.")

    return student_context_similarity