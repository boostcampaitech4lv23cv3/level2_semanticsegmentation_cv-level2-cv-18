# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# # runtime settings
# runner = dict(type='IterBasedRunner', max_iters=40000)
# checkpoint_config = dict(interval=20)
# evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)

lr = 1e-4
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
optimizer_config = dict()

# optimizer_config = dict(type='Fp16OptimizerHook',
# loss_scale = dict(init_scale=1.0,growth_factor=4,backoff_factor=0.5,growth_interval=2000),grad_clip=dict(max_norm=1, norm_type=2))
# fp16 = dict()
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=300,
#     warmup_ratio=1.0 / 10,
#     min_lr_ratio=7e-6)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[15, 19])


# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=1, metric='mIoU',save_best='mIoU')