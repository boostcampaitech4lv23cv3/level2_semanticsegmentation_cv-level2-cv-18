_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/custom_trash_withaug.py', '../_base_/wandb_runtime.py',
    '../_base_/schedules/T4148_schedule_40k.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_tiny_p16_384_20220308-cce8c795.pth'  # noqa

model = dict(
    pretrained=checkpoint,
    backbone=dict(embed_dims=192, num_heads=3),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=192,
        channels=192,
        num_heads=3,
        embed_dims=192))

optimizer = dict(lr=0.001, weight_decay=0.0)

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
crop_size = (512, 512)

data = dict(
    # num_gpus: 8 -> batch_size: 8
    samples_per_gpu=64,
    )
