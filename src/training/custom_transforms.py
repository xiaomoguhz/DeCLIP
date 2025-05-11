import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop, InterpolationMode


class CustomRandomResize(nn.Module):

    def __init__(self, scale=(0.5, 2.0), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.min_scale, self.max_scale = min(scale), max(scale)
        self.interpolation = interpolation

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = random.uniform(self.min_scale, self.max_scale)
        new_size = [int(height * scale), int(width * scale)]
        img = F.resize(img, new_size, self.interpolation)

        return img


class CustomRandomCrop(RandomCrop):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """

        width, height = F.get_image_size(img)
        tar_h, tar_w = self.size

        tar_h = min(tar_h, height)
        tar_w = min(tar_w, width)
        i, j, h, w = self.get_params(img, (tar_h, tar_w))

        return F.crop(img, i, j, h, w)
