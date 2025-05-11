_base_ = './base_config.py'

# model settings
model = dict(
   name_path='configs/cls_voc21.txt',
    type='DeCLIPSegmentation',
    clip_type='EVA02-CLIP-B-16',
     pretrained='eva',
     checkpoint="checkpoints/evab_dinov2B_csa_560_0.25_seg.pt",
     mode="csa",
     prob_thd=0.4,    
    logit_scale=80,
)

# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = 'path_to_your/VOCdevkit/VOC2012'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(4096, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
   dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))