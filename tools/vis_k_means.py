from PIL import Image
from featup.util import UnNormalize
import matplotlib.pyplot as plt
from open_clip.transform import  det_image_transform
import numpy as np
import torch
import os
from tqdm import tqdm
model_name="EVA02-CLIP-B-16" 
# model_name="EVA02-CLIP-L-14-336"
ckpt="logs/exp34/checkpoints/epoch_6.pt"
mean=[0.48145466, 0.4578275, 0.40821073]
std=[0.26862954, 0.26130258, 0.27577711]
val_image_root=""
k_means_root="logs/test"
device="cuda:0"
output_path="test_result"
def visualize_segmentation(image, mask, output_path, alpha=0.5):
    """
    可视化图像的分割掩码，并保存结果到指定位置。
    参数:
    - image: 原图，形状为 (C, H, W)
    - mask: 上采样后的掩码，形状为 (1, 1, H, W)
    - output_path: 输出图像的保存路径
    - alpha: 掩码的透明度
    """
    # 将 mask 转换为 numpy 数组
    mask_np = mask.squeeze().cpu().numpy()  # 形状为 (H, W)
    unnorm = UnNormalize(mean, std)
    image = unnorm(image.unsqueeze(0))[0]

    # 定义颜色映射
    def create_color_map(num_classes):
        colors = plt.get_cmap('jet', num_classes)
        return (colors(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)

    num_classes = int(mask_np.max() + 1)  # 类别数
    color_map = create_color_map(num_classes)

    # 创建一个 RGB 图像来存储掩码的颜色
    mask_color = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)

    for class_id in range(num_classes):
        mask_color[mask_np == class_id] = color_map[class_id]

    # 调整透明度
    image_np = image.permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C)
    image_np = (image_np * 255).astype(np.uint8)  # 假设原图在 [0, 1] 范围内

    # 创建一个透明图像
    overlay = (mask_color * alpha + image_np * (1 - alpha)).astype(np.uint8)

    # 可视化并保存结果
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title('Image with Segmentation Overlay')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__=="__main__":
    preprocess_val_det = det_image_transform(
            256,
            is_train=False,
            mean=mean,
            std=std)
    files=os.listdir(k_means_root)
    k_means_files=[i for i in files if i.endswith(".npy")]
    for file in tqdm(k_means_files, desc="Processing files"):
        image_name=file.replace(".npy",".jpg")
        _image=Image.open(os.path.join(val_image_root,image_name))
        _image=preprocess_val_det(_image).to(device)
        _mask = torch.from_numpy(np.load(os.path.join(k_means_root,file))).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
        _mask=torch.nn.functional.interpolate(_mask,size=(_image.shape[-2], _image.shape[-1]))
        visualize_segmentation(_image,_mask,os.path.join(output_path,image_name))
