from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from open_clip.utils import freeze_batch_norm_2d
from torchvision.ops import roi_align


class FrozenBatchNorm2d(nn.Module):
    _version = 3
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None,
                 freeze_output=True):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.spacial_dim = spacial_dim

        if freeze_output:
            print(f'Freeze the V2L layer', flush=True)
            for p in self.c_proj.parameters():
                p.requires_grad = False
            for p in self.v_proj.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

    def rescale_positional_embedding(self, out_size, dtype):
        h, w = out_size
        rescaled_positional_embedding = \
            self.positional_embedding.new_zeros(1 + h*w, self.positional_embedding.shape[1])
        rescaled_positional_embedding[0] = self.positional_embedding[0]
        pe_2d = self.positional_embedding[1:].T.contiguous().view(
            1, -1, self.spacial_dim, self.spacial_dim)
        pe_2d = F.interpolate(pe_2d, out_size, mode='bicubic', align_corners=False).view(-1, h*w)
        rescaled_positional_embedding[1:] = pe_2d.T.contiguous()

        return rescaled_positional_embedding.to(dtype=dtype)

    def proj_without_attn(self, value):
        value = F.linear(value, self.v_proj.weight, bias=self.v_proj.bias)
        value = F.linear(value, self.c_proj.weight, bias=self.c_proj.bias)

        return value

    def forward_dense(self, x):
        bs, _, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        if h == self.spacial_dim and w == self.spacial_dim:
            pe = self.positional_embedding[:, None, :].to(x.dtype)
        else:
            pe = self.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype)[:, None, :]

        x = x + pe  # (HW+1)NC

        x = self.proj_without_attn(x)

        return x[1:].permute(1, 2, 0).view(bs, -1, h, w)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64,
                 freeze_output=True,
                 freeze_all_bns=True):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size
        self.freeze_output = freeze_output
        self.freeze_all_bns = freeze_all_bns
        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim, freeze_output)
        self.attnpool_input_size = image_size // 32

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def lock(self, unlocked_groups=0, freeze_bn_stats=True):
        assert freeze_bn_stats
        def _lock(module):
            for param in module.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(module)
            module.eval()

        freeze_at = 5 - unlocked_groups
        print(f'Freeze the resnet at {freeze_at}', flush=True)

        if freeze_at >= 1:  # stem
            _lock(self.conv1)
            _lock(self.bn1)
            _lock(self.conv2)
            _lock(self.bn2)
            _lock(self.conv3)
            _lock(self.bn3)
        # each stage is a torch.nn.modules.container.Sequential
        for idx, stage in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], start=2):
            if freeze_at >= idx:
                for block in stage.children():  # each block is a Bottleneck
                    _lock(block)
        if self.freeze_all_bns:
            print(f'Freeze all bn layers', flush=True)           # TODO: study if this is necessary
            freeze_batch_norm_2d(self)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        with torch.no_grad():
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

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

    def _extract_roi_features_v1(self, x, normed_boxes, **kwargs):
        with torch.no_grad():    # TODO: speed up trick
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        x = self.layer4(x)

        tar_size = self.attnpool_input_size
        # TODO: debug
        roi_feats = roi_align(x, self._denormalize_boxes(normed_boxes, x),
                              (tar_size, tar_size), 1.0, -1, True)

        roi_feats = self.attnpool(roi_feats)

        return roi_feats

    def extract_roi_features(self, x, normed_boxes, extract_type='v1'):
        if extract_type == 'v1':
            return self._extract_roi_features_v1(x, normed_boxes)
        else:
            assert extract_type == 'v2'
            return self._extract_roi_features_v2(x, normed_boxes)

    def mask_attn_pool(self, image, masks):
        return self.mask_pool(image, masks)

    def mask_pool(self, image, masks):
        x = self.stem(image)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feature_map = self.attnpool.forward_dense(x)
        feature_map = F.normalize(feature_map, dim=1)          # remember to normalize!

        feature_map = feature_map.flatten(-2, -1)   # bs, c, h*w
        num_masks_per_image = [len(masks_per_image) for masks_per_image in masks]
        masks = torch.cat(masks).float().flatten(-2, -1)    # bs, h*w
        feature_map = torch.repeat_interleave(
            feature_map, torch.tensor(num_masks_per_image, device=feature_map.device), dim=0)
        features = (feature_map * masks[:, None]).sum(-1) / (masks.sum(1, keepdim=True) + 1e-12)

        return features

    def _extract_roi_features_v2(self, x, normed_boxes, **kwargs):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.attnpool.forward_dense(x)
        x = F.normalize(x, dim=1)          # remember to normalize!
        # TODO: debug
        roi_feats = roi_align(x, self._denormalize_boxes(normed_boxes, x),
                              (1, 1), 1.0, -1, True)[:, :, 0, 0]
        return roi_feats
    # def _extract_roi_features_v2(self, x, normed_boxes, **kwargs):
    #     with torch.no_grad():   # TODO speed up trick
    #         x = self.stem(x)
    #         x = self.layer1(x)
    #         x = self.layer2(x)
    #         x = self.layer3(x)
    #     tar_size = self.attnpool_input_size * 2
    #     # TODO: debug
    #     roi_feats = roi_align(x, self._denormalize_boxes(normed_boxes, x),
    #                           (tar_size, tar_size), 1.0, -1, True)
    #
    #     roi_feats = self.layer4(roi_feats)
    #     roi_feats = self.attnpool(roi_feats)
    #
    #     return roi_feats

    def encode_dense(self, x, keep_shape=True):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feature_map = self.attnpool.forward_dense(x)
        feature_map = F.normalize(feature_map, dim=1)  # remember to normalize!

        return feature_map
