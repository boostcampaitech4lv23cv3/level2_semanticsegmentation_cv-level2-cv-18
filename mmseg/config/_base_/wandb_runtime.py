# yapf:disable
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='MMSegWandbHook',
                     init_kwargs={
                         'entity': "light-observer",
                         'project': "Trash_MMseg",
                         "name": "segformer_mit-b0"
                     },
                     interval=200,
                     log_checkpoint=True,
                     log_checkpoint_metadata=True,
                     num_eval_images=10)
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
