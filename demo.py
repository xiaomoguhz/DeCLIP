from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from declip_segmentor import DeCLIPSegmentation
import torch
import os

device="cuda:0"
img = Image.open('demo_images/whitedog.jpg')
output_dir="demo_vis"
name_list = ['dog', 'grass']
checkpoint="checkpoints/evab_dinov2B_csa_560_0.25_seg.pt"

with open('configs/demo_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list) - 1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
writers.close()

with torch.no_grad():
    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])])(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    model = DeCLIPSegmentation(clip_type='EVA02-CLIP-B-16',
                                    pretrained='eva',
                                    checkpoint=checkpoint,
                                    mode="csa",
                                name_path='configs/demo_name.txt',).to(device)
    seg_pred = model.predict(img_tensor, data_samples=None)
    seg_pred = seg_pred.data.cpu().numpy().squeeze(0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(seg_pred, cmap='viridis')
    ax[1].axis('off')
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir,'seg_pred.png'), bbox_inches='tight')

