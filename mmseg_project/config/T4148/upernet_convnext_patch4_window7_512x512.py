_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/custom_trash.py',
    '../_base_/wandb_runtime.py', '../_base_/schedules/T4148_schedule_40k.py'
]

data = dict(
    samples_per_gpu=8)