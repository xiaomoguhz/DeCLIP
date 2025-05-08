data_root=""  # path to your coco dataset
pretrain_ckpt=""  # path to your EVA-CLIP checkpoint
exp_name=""  # output folder name
vfm_type=dinov2-L # {sam-B, sam-L, dinov2-B, dinov2-L, dino-B-8, dino-B-16}

# always keep total batchsize=16 , otherwise, Linear scaling the learning rate
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29500 -m training.main --batch-size=2 --lr=1e-5 --wd=0.1 --epochs=6 --workers=4 \
--model EVA02-CLIP-L-14-336 --pretrained eva --warmup 1000  --zeroshot-frequency 1 --dataset-type grid_distill  \
--test-type coco_panoptic --train-data ${data_root}/annotations/instances_train2017.json \
--val-data ${data_root}/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy --train-image-root ${data_root}/train2017 \
--val-image-root ${data_root}/val2017 --cache-dir ${pretrain_ckpt} --log-every-n-steps 100 \
--lock-image --save-frequency 1 --lock-image-unlocked-groups 24 \
--name ${exp_name} --downsample-factor 14 --det-image-size 560 --val-segm-root  ${data_root}/annotations/panoptic_val2017 \
--alpha 0.95  --use_vfm ${vfm_type} --mode qq_vfm_distill --loss_context_weight 0.1 --loss_content_weight 1.0