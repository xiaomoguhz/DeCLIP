import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def get_fed_loss_inds(gt_classes, num_sample_cats, C):
    appeared = torch.unique(gt_classes) # C'
    prob = appeared.new_ones(C).float()
    if len(appeared) < num_sample_cats:
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared),
            replacement=False)
        appeared = torch.cat([appeared, more_appeared])
    return appeared


class RegionCLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        embed_path = args.train_embed_path
        noun_embeddings = torch.from_numpy(np.load(embed_path))
        noun_embeddings = F.normalize(noun_embeddings, dim=-1)
        self.register_buffer("noun_embeddings", noun_embeddings)
        self.place_holder = nn.Parameter(torch.ones(1))

    def __call__(self, batch, model, dist_model, loss, device, cast_dtype,
                 distributed, args):
        if distributed:
            model = model.module
        images, boxes = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        boxes = boxes.to(device=device, non_blocking=True)

        boxes_list = []
        boxes_label_list = []

        for boxes_per_image in boxes:
            boxes_per_image = boxes_per_image[boxes_per_image[:, -1] > 0.5]
            boxes_label_list.append(boxes_per_image[:, 4].long())
            boxes_list.append(boxes_per_image[:, :4])
        boxes_labels = torch.cat(boxes_label_list)
        box_features = model.encode_pseudo_boxes(images, boxes_list, normalize=True,
                                                 extract_type=args.extract_type)
        temp = model.logit_scale.exp().detach()
        boxes2nouns = box_features @ self.noun_embeddings.T * temp
        target = torch.zeros_like(boxes2nouns)
        target[range(len(boxes_labels)), boxes_labels] = 1.0

        appeared = get_fed_loss_inds(boxes_labels, 100, self.noun_embeddings.shape[0])
        target = target[:, appeared]
        boxes2nouns = boxes2nouns[:, appeared]

        loss_cls = F.binary_cross_entropy_with_logits(boxes2nouns, target, reduction='none')  # B x C
        loss_cls = loss_cls.sum(-1).mean()

        image_size = model.visual.image_size
        if isinstance(image_size, int):
            tar_h = tar_w = image_size
        else:
            tar_h, tar_w = image_size
        images = F.interpolate(images, size=(tar_h, tar_w), mode='bilinear')

        losses = dict(loss_contrast=loss_cls * args.contrast_weight)

        return losses, len(images), temp
