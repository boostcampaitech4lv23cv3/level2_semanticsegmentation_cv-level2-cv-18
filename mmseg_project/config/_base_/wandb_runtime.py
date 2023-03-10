# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='MMSegWandbHook',
                     init_kwargs={
                         'entity': "light-observer",
                         'project': "Trash_MMseg",
                         "name": "segmaster_vit_small"
                     },
                    num_eval_images=0)
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
