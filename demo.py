import torch
from PIL import Image
from src.open_clip.factory import create_model
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, Resize

if __name__ == "__main__":
    with torch.no_grad():
        img_path = "path_to_your_img"
        device = "cuda:0"
        output_dir = "vis"
        model = create_model("EVA02-CLIP-L-14-336","eva",precision="fp32",device=device,pretrained_image=False,pretrained_hf=True,cache_dir="path_to_your_trained_declip",).eval().to(device)
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        normalize_transform = Normalize(mean=mean, std=std)
        custom_transform = Compose([Resize((336, 336)), CenterCrop(336), ToTensor(), normalize_transform])
        raw_img = Image.open(img_path).convert("RGB")
        img_tensor = custom_transform(raw_img).to(device).unsqueeze(0)
        model_feat2 = model.encode_dense(img_tensor, normalize=True, keep_shape=True, mode="qq") # w/ final layernorm & v-l projection layer
        print(model_feat2.shape)