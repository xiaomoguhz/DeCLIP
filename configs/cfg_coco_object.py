_base_ = './base_config.py'

# model settings
model = dict(
    name_path='configs/cls_coco_object.txt',
    type='DeCLIPSegmentation',
    clip_type='EVA02-CLIP-B-16',
     pretrained='eva',
     checkpoint="checkpoints/evab_dinov2B_csa_560_0.25_seg.pt",
    mode="csa",
    prob_thd=0.2,
)

# dataset settings
dataset_type = 'COCOObjectDataset'
data_root = 'path_to_your/coco_obj'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
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
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))