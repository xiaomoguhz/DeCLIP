import torch
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.open_clip.factory import create_model
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, Resize
import io

def process_image(image):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    model = create_model(
        "ViT-L-14-336",
        "openai",
        precision="fp32",
        device=device,
        pretrained_image=True,
        pretrained_hf=True,
        cache_dir=None
    ).eval().to(device)
    
    # Define transforms
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    normalize_transform = Normalize(mean=mean, std=std)
    custom_transform = Compose([
        Resize((336, 336)),
        CenterCrop(336),
        ToTensor(),
        normalize_transform
    ])
    
    # Process image
    with torch.no_grad():
        img_tensor = custom_transform(image).to(device).unsqueeze(0)
        model_feat = model.encode_dense(img_tensor, normalize=True, keep_shape=True, mode="qq")
        
    # Convert feature tensor to numpy for visualization
    feature_map = model_feat[0].mean(dim=0).cpu().numpy()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot feature map
    im = ax2.imshow(feature_map, cmap='viridis')
    ax2.set_title('Feature Map')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2)
    
    # Convert plot to PIL Image
    plt.tight_layout()
    
    # Save the figure to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    feature_vis = Image.open(buf)
    plt.close(fig)
    
    # Create detailed shape information
    shape_info = f"Feature Map Shape: {model_feat.shape}\n"
    shape_info += f"Number of channels: {model_feat.shape[1]}\n"
    shape_info += f"Spatial dimensions: {model_feat.shape[2]}x{model_feat.shape[3]}"
    
    return shape_info, feature_vis

# Create Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Model Output Information"),
        gr.Image(label="Visualization")
    ],
    title="CLIP Image Feature Extractor",
    description="Upload an image to extract features using the ViT-L-14-336 model. The visualization shows the original image and the corresponding feature map."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)