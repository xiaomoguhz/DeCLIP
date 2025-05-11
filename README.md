# DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception
This branch of the repository is the official PyTorch implementation of [DeCLIP](https://arxiv.org/abs/2505.04410) for zero-shot image segmentation.


## Contributions
<p align="center">
  <img src="assets/problem.png" alt="Problem Analysis" width="550">
  <img src="assets/performance.png" alt="Performance Comparison" width="230" >
</p>

1. We analyze CLIP and find that its limitation in open-vocabulary dense prediction arises from **image tokens failing to aggregate information from spatially or semantically related regions**.
2. To address this issue, we propose DeCLIP, a simple yet effective unsupervised fine-tuning framework, to enhance the discriminability and spatial consistency of CLIP’s local features via **a decoupled feature enhancement strategy**.
3. DeCLIP outperforms previous state-of-the-art models on a broad range of open-vocabulary dense prediction benchmarks.

## 🌈Environment
- Linux with Python == 3.10.0
- CUDA 11.7
- The provided environment is suggested for reproducing our results, similar configurations may also work.

## 🚀Quick Start

#### 1. Create Conda Environment
```
conda create -n DeCLIP_ZSSS python=3.10.0
conda activate DeCLIP_ZSSS
pip install -r requirements.txt
pip install -e . -v
```
#### 2. Install MMCV
```
# Replace {cu_version} and {torch_version} with your CUDA and PyTorch versions
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html

# For example, if using CUDA 11.7 and PyTorch 2.0:
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
```

#### 3. Dataset Preparation
We follow previous works for preparing datasets. Thanks to [SCLIP](https://github.com/wangf3014/SCLIP), [ClearCLIP](https://github.com/mc-lan/ClearCLIP/tree/main), and [ProxyCLIP](https://github.com/mc-lan/ProxyCLIP/tree/main) for their open-source contributions, as detailed below:
We include the following dataset configurations in this repo: 
1) `With background class`: PASCAL VOC (21), PASCAL Context (60), Cityscapes, ADE20k, and COCO-Stuff164k, 
2) `Without background class`: VOC20, Context59 (i.e., PASCAL VOC and PASCAL Context without the background category), and COCO-Object.

Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets. 

The COCO-Object dataset can be converted from COCO-Stuff164k by executing the following command:
```
python datasets/cvt_coco_object.py PATH_TO_COCO_STUFF164K -o PATH_TO_COCO164K
```

#### 4. Evaluation

Please download our pre-trained DeCLIP weights from this [Link](https://huggingface.co/xiaomoguhzz/DeCLIP_evab_dinov2B_csa_560_0.25_seg) and prepare it as follows:
```text
DeCLIP_ZSSS/
├── checkpoints
    ├── DeCLIP_evab_dinov2B_csa_560_0.25_seg
```

#### 5. Modify the Necessary Parameters.
Before starting the evaluation, please modify the paths in the config file, for example, in `configs/cfg_ade20k.py`:
```
# model settings
model = dict(
    name_path='configs/cls_ade20k.txt',
    type='DeCLIPSegmentation',
    clip_type='EVA02-CLIP-B-16',
     pretrained='eva',
    checkpoint="checkpoints/evab_dinov2B_csa_560_0.25_seg.pt",
    mode="csa",
)

# dataset settings
dataset_type = 'ADE20KDataset'
data_root = 'path_to_your/ADEChallengeData2016'

``` 
After modifying the dataset path, run the script to verify DeCLIP's performance on ADE20K.

``` 
CUDA_VISIBLE_DEVICES=0 python eval.py --config configs/cfg_ade20k.py --work-dir logs/ade20k
``` 
The verification of other datasets is the same; you only need to modify the different `--config` and `--work-dir`.

If you are interested in visualizing the inference results, you can easily do so through the mmseg interface by modifying the `configs/base_config.py` file as follows:
``` 
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', 
                       draw=True, # change this to True
                       interval=100))
``` 
Then the reasoning visualization will be saved in the  `--work-dir` folder. 
#### 5. Quickly experience DeCLIP's zero-shot image segmentation capability.
``` 
python demo.py
``` 

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