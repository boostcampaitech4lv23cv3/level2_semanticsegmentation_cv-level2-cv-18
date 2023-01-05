_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/custom_trash.py',
    '../_base_/wandb_runtime.py', '../_base_/schedules/T4148_schedule_40k.py'
]

model = dict(
    pretrained='/opt/ml/trash/pretrain/beit_base_modelpreT.pth',
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(426, 426)))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

lr_config = dict(
    _delete_=True,
    
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=8)
