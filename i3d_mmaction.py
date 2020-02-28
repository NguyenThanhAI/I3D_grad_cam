import os
import argparse
import mmcv

import torch

from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.runner import get_dist_info
from mmcv.parallel import scatter, collate, MMDataParallel, MMDistributedDataParallel

from mmaction import datasets
from mmaction.apis import init_dist
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer, recognizers
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy, mean_class_accuracy)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, default="i3d_kinetics400_3d_rgb_r50_c3d_inflate3x1x1_seg1_f32s2.py", help="test config file path")
parser.add_argument("--checkpoint", type=str, default="i3d_kinetics_rgb_r50_c3d_inflated3x1x1_seg1_f32s2_f32s2-b93cc877.pth", help="checkpoint file")

args = parser.parse_args()

cfg = mmcv.Config.fromfile(args.config)

cfg.data.test.test_mode = True

if cfg.data.test.oversample == 'three_crop':
    cfg.model.spatial_temporal_module.spatial_size = 8

model = build_recognizer(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)