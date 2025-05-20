# DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception
This repository is the official PyTorch implementation of [DeCLIP](https://arxiv.org/abs/2505.04410). 



## Overview
![Alt text](assets/DeCLIP.png)
DeCLIP is an unsupervised fine-tuning framework for open-vocabulary dense perception tasks, which decouples CLIP's self-attention module to obtain "content" and "context" features, learning from itself and vision foundation models respectively to enhance local discriminability and spatial consistency.
## Contributions


<p align="center">
  <img src="assets/problem.png" alt="Problem Analysis" width="550">
  <img src="assets/performance.png" alt="Performance Comparison" width="230" >
</p>

1. We analyze CLIP and find that its limitation in open-vocabulary dense prediction arises from **image tokens failing to aggregate information from spatially or semantically related regions**.
2. To address this issue, we propose DeCLIP, a simple yet effective unsupervised fine-tuning framework, to enhance the discriminability and spatial consistency of CLIP’s local features via **a decoupled feature enhancement strategy**.
3. DeCLIP outperforms previous state-of-the-art models on a broad range of open-vocabulary dense prediction benchmarks.

## Video
<p align="center" width="80%">
  <a href="https://youtu.be/hr2OEZL5Kgs?si=EhzALIpGmLgQyVbB" target="_blank">
    <img src="assets/cover.png" alt="DeCLIP Video" style="width: 80%; min-width: 300px; display: block; margin: auto;">
  </a>
</p>

## 🎉News
* **[2025.05.07]**  We will update the complete training and inference code as well as weights. Stay tuned!
* **[2025.02.27]**  Our work has been accepted at CVPR 2025.

## 🔥TODO
- [x] Initialize Project
- [x] Release the training and inference code of DeCLIP
- [x] Release evaluation code for DeCLIP in open vocabulary semantic segmentation based on VLM features. Please refer to the [ZSSS branch](https://github.com/xiaomoguhz/DeCLIP/tree/ZSSS).
- [x] Release the code to integrate DeCLIP into CAT-Seg. Please refer to the [DeCLIP_CATSeg](https://github.com/xiaomoguhz/DeCLIP/tree/DeCLIP_CATSeg).
- [ ] Release the code to integrate DeCLIP into F-ViT and OV-DQUO
## 🌈Environment
- Linux with Python == 3.10.0
- CUDA 11.7
- The provided environment is suggested for reproducing our results, similar configurations may also work.

## 🚀Quick Start

#### 1. Create Conda Environment
```
conda create -n DeCLIP python=3.10.0
conda activate DeCLIP
pip install -r requirements.txt
pip install -e . -v
```
#### 2. Dataset Preparation
The distillation process of DeCLIP does not rely on any annotations and only requires input images. In our paper, the [COCO](https://cocodataset.org/#home)  dataset was used. Please download the dataset and organize the folders as follows.
```text
DeCLIP/
├── dataset
    ├── coco
        ├── annotations
            ├── instances_train2017.json  # only access images
            ├── panoptic_val2017.json # for validation
            ├── panoptic_val2017     
        ├── train2017
        ├── val2017
```
#### 3.Preparing Pretrained Checkpoints
Please download the pretrained weights from [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP) and organize them as shown below.
```text
DeCLIP/
├── checkpoints
    ├── EVA02_CLIP_B_psz16_s8B.pt
    ├── EVA02_CLIP_L_336_psz14_s6B.pt
```
#### 4. Modify the Necessary Parameters.
Before starting the training, please modify the paths in the training script.
```
data_root=** # path to your coco dataset
pretrain_ckpt=**  # path to your EVA-CLIP checkpoint
exp_name=**  # output folder name
vfm_type=**  # which vfm to use, we use dinov2-B & dinov2-L for default, {sam-B, sam-L, dinov2-B, dinov2-L, dino-B-8, dino-B-16}
``` 
> **Note:** If you encounter any freezing or hanging issues during training, it is likely due to problems with downloading or loading the Visual Feature Models (VFMs) from torch hub. We highly recommend manually downloading the SAM, DINOV2, or DINO code and weights locally before proceeding with DeCLIP training.  
> You can modify the relevant code in the [build_vfm()](src/training/build_vfm) located in `DeCLIP/src/training`.
#### 5. Script for training DeCLIP
To train the DeCLIP on the COCO dataset, please run the following script:
``` 
# dist training, EVA-CLIP, ViT-B-16
bash scripts/dist_DeCLIP_eva_vitb16_coco.sh

# dist training, EVA-CLIP, ViT-L-16-336
bash scripts/dist_DeCLIP_eva_vitL14_336_coco.sh
```
### 5. Use DeCLIP for zero-shot image segmentation.
Please refer to the [ZSSS branch](https://github.com/xiaomoguhz/DeCLIP/tree/ZSSS).

### 6. Using DeCLIP for open-vocabulary semantic segmentation in the CAT-Seg model.
Please refer to the [DeCLIP_CATSeg](https://github.com/xiaomoguhz/DeCLIP/tree/DeCLIP_CATSeg).
## ❤️ Acknowledgement
Our work builds upon the method and codebase of [CLIPSelf](https://github.com/wusize/CLIPSelf), [ClearCLIP](https://github.com/mc-lan/ClearCLIP), [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg), [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/v2.16.0). We sincerely thank the authors for their remarkable contributions, which provided an essential foundation for our research.

## 🙏 Citing DeCLIP 

```bibtex
@article{wang2025declip,
  title={DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception},
  author={Wang, Junjie and Chen, Bin and Li, Yulin and Kang, Bin and Chen, Yichi and Tian, Zhuotao},
  journal={arXiv preprint arXiv:2505.04410},
  year={2025}
}
```