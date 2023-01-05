# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='MMSegWandbHook',
                     init_kwargs={
                         'entity': "light-observer",
                         'project': "Trash_MMseg",
                         "name": "uper_beit_large_backboneTrain_withaug"
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
