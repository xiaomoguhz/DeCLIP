import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms.v2 import ScaleJitter
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop
from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
import numpy as np
@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float]]] = None
    interpolation: Optional[str] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False


class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill
        
    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        new_size = tuple(round(dim * scale) for dim in (height, width))
        img = F.resize(img, new_size, self.interpolation)
        pad_h = self.max_size - new_size[0]
        pad_w = self.max_size - new_size[1]
        img = F.pad(img, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=self.fill)

        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()
    normalize = Normalize(mean=mean, std=std)
    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        use_timm = aug_cfg_dict.pop('use_timm', False)
        if use_timm:
            from timm.data import create_transform  # timm can still be optional
            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                input_size = (3,) + image_size[-2:]
            else:
                input_size = (3, image_size, image_size)
            # by default, timm aug randomly alternates bicubic & bilinear for better robustness at inference time
            aug_cfg_dict.setdefault('interpolation', 'random')
            aug_cfg_dict.setdefault('color_jitter', None)  # disable by default
            train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                hflip=0.,
                mean=mean,
                std=std,
                re_mode='pixel',
                **aug_cfg_dict,
            )
        else:
            train_transform = Compose([
                RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.pop('scale'),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])
            if aug_cfg_dict:
                warnings.warn(f'Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())}).')
        return train_transform
    else:
        if resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)


def det_image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        fill_color: int = 0,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        raise NotImplementedError
    else:
        transforms = [
            ResizeLongest(image_size, fill=fill_color),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ]
        return Compose(transforms)


class ResizeLongest(nn.Module):
    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[1:]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        new_height, new_width = round(height * scale), round(width * scale)
        img = F.resize(img, [new_height, new_width], self.interpolation, antialias=None)
        pad_h = self.max_size - new_height
        pad_w = self.max_size - new_width
        img = F.pad(img, padding=[0, 0, pad_w, pad_h], fill=self.fill)
        return img


def get_scale(img, new_image):
    if isinstance(img, torch.Tensor):
        height, width = new_image.shape[-2:]
    else:
        width, height = img.size

    if isinstance(new_image, torch.Tensor):
        new_height, new_width = new_image.shape[-2:]
    else:
        new_width, new_height = new_image.size

    scale = min(new_height/height, new_width/width)

    return scale


class FixedSizeCrop:
    """
    If `crop_size` is smaller than the input image size, then it uses a random crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the right and the bottom of the image to the crop size if `pad` is True, otherwise
    it returns the smaller image.
    """
    def __init__(self, crop_size, pad=True, pad_value=128.0, seg_pad_value=255,return_param=False):
        """
        Args:
            crop_size: target image (height, width).
            pad: if True, will pad images smaller than `crop_size` up to `crop_size`
            pad_value: the padding value to the image.
            seg_pad_value: the padding value to the segmentation mask.
        """
        self.crop_size = crop_size  # (height, width)
        self.pad = pad
        self.pad_value = pad_value
        self.seg_pad_value = seg_pad_value
        self.return_param=return_param

    def _get_random_crop_params(self, img, output_size):
        """ Get parameters for a random crop. """
        w, h = img.size # PIL image size is (width, height)
        crop_h, crop_w = output_size
        # If image is larger than the crop size, calculate the random crop parameters
        if h > crop_h and w > crop_w:
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
        else:
            # If the image is smaller, no crop is needed (padding will be applied later if required)
            top = 0
            left = 0
        return top, left, crop_h, crop_w

    def _pad_if_needed(self, img):
        """ Pad the image on the right and bottom if its size is smaller than `crop_size`. """
        w, h = img.size  # PIL image size is (width, height)
        crop_h, crop_w = self.crop_size
        # Calculate required padding for height and width
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        # Only pad if necessary
        if pad_h > 0 or pad_w > 0:
            # Padding order: [left, top, right, bottom]
            img = F.pad(img, padding=[0, 0, pad_w, pad_h], fill=self.pad_value)
        return img

    def __call__(self, img, param=None):
        """ Apply the crop or padding to the image. """
        # First, apply padding if needed (if the image is smaller than the crop size)
        img = self._pad_if_needed(img)
        # Now, the image size is guaranteed to be at least as large as the target crop size
        w, h = img.size
        if param:
            h_scale, w_scale = param
            crop_h, crop_w = self.crop_size
            top, left=int(h_scale*h),int(w_scale*w)

        else:
            top, left, crop_h, crop_w = self._get_random_crop_params(img, self.crop_size)
        # Apply random crop
        img = F.crop(img, top=top, left=left, height=crop_h, width=crop_w)
        if self.return_param:
            return img, (top/h,left/w)
        else:
            return img

class ImgRescale:
    def __init__(self,
                 max_size: Optional[Union[int, Tuple[int, int]]] = (1024, 1024),
                 interpolation=InterpolationMode.BICUBIC):
        """
        Args:
            max_size (Union[int, Tuple[int, int]]): 最大宽度和高度。如果是整数，则表示正方形的最大尺寸。
            interpolation (str): 插值方式，默认为 'bicubic'。
        """
        if isinstance(max_size, int):
            self.max_size = (max_size, max_size)  # 如果提供的是单个整数，则假定宽高相同
        else:
            self.max_size = max_size  # 否则使用提供的 (height, width)
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image or torch.Tensor): 输入的图像。

        Returns:
            img: 调整大小后的图像。
        """
        # 获取图像的宽高
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]  # 如果是 Tensor，形状为 (C, H, W)
        else:
            width, height = img.size  # 如果是 PIL.Image，获取图像的宽高
        max_long_edge = max(self.max_size)
        max_short_edge = min(self.max_size)
        scale_factor = min(max_long_edge / max(height, width),
                           max_short_edge / min(height, width))
        # 计算新的尺寸
        new_size = (round(height * scale_factor), round(width * scale_factor))

        # 调整图像大小
        img = F.resize(img, new_size, self.interpolation)
        
        return img
