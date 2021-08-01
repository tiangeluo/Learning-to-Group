#!/usr/bin/env python
"""Test point cloud part segmentation models"""

from __future__ import division
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(__file__) + '/..')

import argparse
import logging
import time
import json
from collections import defaultdict, OrderedDict
import copy

import numpy as np
from geometry_utils import *

from core.config import purge_cfg
from core.utils.logger import setup_logger
from IPython import embed

import time
from subprocess import call
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Learning Test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

def preprocess(data_dict):
    data_dict = {k: np.array(v) for k, v in data_dict.items()}
    data_dict['points'] = data_dict['points'].transpose()

    return data_dict

def visualize(cfg, output_dir=''):

    output_dir = osp.join(output_dir, 'tree')
    save_shape_dir = osp.join(output_dir, 'shapes')
    os.makedirs(save_shape_dir, exist_ok=True)
    png_dir = osp.join(output_dir, 'png')
    os.makedirs(png_dir, exist_ok=True)
    html_dir = osp.join(output_dir, 'html')
    os.makedirs(html_dir, exist_ok=True)

    shape_list = []
    for root, _, files in os.walk(save_shape_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
            file_path = osp.join(root, file)
            with open(file_path) as f:
                shape_list.append((file_path,
                                   preprocess(json.load(f))))

    # uncomment the below lines can specify shape.
    #target_shape_id = int(input('target_shape_id'))
    #for shape_id, (file_path, out_dict) in enumerate(sorted(shape_list, key=lambda x: x[0])):
    #    if shape_id < target_shape_id:
    #        continue
    #    if shape_id == 40:
    #        break
    for shape_id, (file_path, out_dict) in enumerate(sorted(shape_list, key=lambda x: x[0])):
        print(shape_id)
        points = out_dict['points']
        num_points = points.shape[0]
        mask_pool = out_dict['mask_pool']
        cur_png_dir = osp.join(png_dir, '%03d'%shape_id)
        os.makedirs(cur_png_dir, exist_ok=True)
        for i in range(mask_pool.shape[0]):
            render_pts_with_label(os.path.join(cur_png_dir, str(i)+'.png'), points, mask_pool[i].astype(np.int32))
        cmd = 'python partnet/visu_tree_html.py %s %s' % (output_dir,'%03d'%shape_id)
        call(cmd, shell=True)

def main():
    args = parse_args()

    # Load the configuration
    from partnet.config.ins_seg_3d import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # Replace '@' with config path
    if output_dir:
        output_dir = './results/'+cfg.DATASET.PartNetInsSeg.TEST.shape
        output_dir = osp.join(output_dir,'Level_%d'%cfg.TEST.LEVEL)

    logger = setup_logger('shaper', output_dir, prefix='test')
    #logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.TASK == 'ins_seg_3d'
    visualize(cfg, output_dir)

    cmd = 'cp index.html %s'%(os.path.join(output_dir,'tree','html'))
    call(cmd, shell=True)

if __name__ == '__main__':
    main()
