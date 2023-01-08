# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint, wrap_fp16_model
from PIL import Image

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
import cv2
from pycocotools.coco import COCO
import pandas as pd


@torch.no_grad()
def main(args):

    models = []
    gpu_ids = args.gpus
    configs = args.config
    ckpts = args.checkpoint

    cfg = mmcv.Config.fromfile(configs[0])

    if args.aug_test:
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
        ]
        cfg.data.test.pipeline[1].flip = True
    else:
        cfg.data.test.pipeline[1].img_ratios = [1.0]
        cfg.data.test.pipeline[1].flip = False

    torch.backends.cudnn.benchmark = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False,
    )

    for idx, (config, ckpt) in enumerate(zip(configs, ckpts)):
        cfg = mmcv.Config.fromfile(config)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        if cfg.get('fp16', None):
            wrap_fp16_model(model)
        load_checkpoint(model, ckpt, map_location='cpu')
        torch.cuda.empty_cache()
        tmpdir = args.save_dir
        mmcv.mkdir_or_exist(tmpdir)
        model = MMDataParallel(model, device_ids=[gpu_ids[idx % len(gpu_ids)]])
        model.eval()
        models.append(model)

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler

    coco = COCO('/opt/ml/input/data/test.json')
    prediction_strings = []
    file_names = []
    for batch_indices, data in zip(loader_indices, data_loader):
        result = []
        for model in models:
            x, _ = scatter_kwargs(
                inputs=data, kwargs=None, target_gpus=model.device_ids)
            if args.aug_test:
                logits = model.module.aug_test_logits(**x[0])
            else:
                logits = model.module.simple_test_logits(**x[0])
            result.append(logits)

        result_logits = 0
        for logit in result:
            result_logits += logit

        pred = result_logits.argmax(axis=1).squeeze()
        out = np.array(pred)

        image_info = coco.loadImgs(coco.getImgIds(imgIds=batch_indices[0]))[0]
        out = cv2.resize(out, (256,256), interpolation =cv2.INTER_NEAREST)
        out = out.flatten().astype(int)
        prediction_string = ' '.join(str(o) for o in out.tolist())
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
        prog_bar.update()
    
    submission = pd.DataFrame()
    submission['image_id'] = file_names
    submission['PredictionString'] = prediction_strings
    csv_path = os.path.join(tmpdir, 'ensemble.csv')
    submission.to_csv(csv_path, index=False)
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='Model Ensemble with logits result')
    parser.add_argument(
        '--config', type=str, nargs='+', help='ensemble config files path')
    parser.add_argument(
        '--checkpoint',
        type=str,
        nargs='+',
        help='ensemble checkpoint files path')
    parser.add_argument(
        '--aug-test',
        action='store_true',
        help='control ensemble aug-result or single-result (default)')
    parser.add_argument(
        '--save-dir', type=str, default='results', help='the dir to save result')
    parser.add_argument(
        '--gpus', type=int, nargs='+', default=[0], help='id of gpu to use')

    args = parser.parse_args()
    assert len(args.config) == len(args.checkpoint), \
        f'len(config) must equal len(checkpoint), ' \
        f'but len(config) = {len(args.config)} and' \
        f'len(checkpoint) = {len(args.checkpoint)}'

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
