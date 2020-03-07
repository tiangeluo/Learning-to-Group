#!/usr/bin/env python
"""Test point cloud classification models"""

from __future__ import division
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(__file__) + '/..')

import argparse
import logging
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from core.config import purge_cfg
from core.utils.checkpoint import Checkpointer
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.torch_util import set_random_seed

from shaper.models.build import build_model
from shaper.data.build import build_dataloader, parse_augmentations
from shaper.data import transforms as T


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


class Evaluator(object):
    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(class_names)

        # The number of true positive
        self.num_tp_per_class = [0 for _ in range(self.num_classes)]
        # The number of ground_truth
        self.num_gt_per_class = [0 for _ in range(self.num_classes)]

    def update(self, pred_label, gt_label):
        pred_label = int(pred_label)
        gt_label = int(gt_label)
        assert 0 <= gt_label < self.num_classes
        if gt_label == pred_label:
            self.num_tp_per_class[gt_label] += 1
        self.num_gt_per_class[gt_label] += 1

    def batch_update(self, pred_labels, gt_labels):
        assert len(pred_labels) == len(gt_labels)
        for pred_label, gt_label in zip(pred_labels, gt_labels):
            self.update(pred_label, gt_label)

    @property
    def overall_accuracy(self):
        return sum(self.num_tp_per_class) / sum(self.num_gt_per_class)

    @property
    def class_accuracy(self):
        acc_per_class = []
        for ind, class_name in enumerate(self.class_names):
            if self.num_gt_per_class[ind] == 0:
                acc = float('nan')
            else:
                acc = self.num_tp_per_class[ind] / self.num_gt_per_class[ind]
            acc_per_class.append(acc)
        return acc_per_class

    def print_table(self):
        from tabulate import tabulate
        table = []
        header = ['Class', 'Accuracy', 'Correct', 'Total']
        acc_per_class = self.class_accuracy
        for ind, class_name in enumerate(self.class_names):
            table.append([class_name, '{:.2f}'.format(100.0 * acc_per_class[ind]),
                          self.num_tp_per_class[ind], self.num_gt_per_class[ind]])
        return tabulate(table, headers=header, tablefmt='psql')

    def print_table(self):
        from tabulate import tabulate
        table = []
        header = ['Class', 'Accuracy', 'Correct', 'Total']
        acc_per_class = self.class_accuracy
        for ind, class_name in enumerate(self.class_names):
            table.append([class_name, 100.0 * acc_per_class[ind],
                          self.num_tp_per_class[ind], self.num_gt_per_class[ind]])
        return tabulate(table, headers=header, tablefmt='psql', floatfmt='.2f')

    def save_table(self, filename):
        from tabulate import tabulate
        header = ['overall acc', 'class acc'] + self.class_names
        acc_per_class = self.class_accuracy
        table = [[self.overall_accuracy, np.nanmean(acc_per_class)] + acc_per_class]
        with open(filename, 'w') as f:
            # In order to unify format, remove all the alignments.
            f.write(tabulate(table, headers=header, tablefmt='tsv', floatfmt='.5f',
                             numalign=None, stralign=None))


def test(cfg, output_dir=''):
    logger = logging.getLogger('shaper.test')

    # build model
    model, loss_fn, metric = build_model(cfg)
    model = nn.DataParallel(model).cuda()
    # model = model.cuda()

    # build checkpointer
    checkpointer = Checkpointer(model, save_dir=output_dir, logger=logger)

    if cfg.TEST.WEIGHT:
        # load weight if specified
        weight_path = cfg.TEST.WEIGHT.replace('@', output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer.load(None, resume=True)

    # build data loader
    test_dataloader = build_dataloader(cfg, mode='test')
    test_dataset = test_dataloader.dataset

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #
    model.eval()
    loss_fn.eval()
    metric.eval()
    set_random_seed(cfg.RNG_SEED)
    evaluator = Evaluator(test_dataset.class_names)

    if cfg.TEST.VOTE.NUM_VOTE > 1:
        # remove old transform
        test_dataset.transform = None
        if cfg.TEST.VOTE.TYPE == 'AUGMENTATION':
            tmp_cfg = cfg.clone()
            tmp_cfg.defrost()
            tmp_cfg.TEST.AUGMENTATION = tmp_cfg.TEST.VOTE.AUGMENTATION
            transform = T.Compose([T.ToTensor()] + parse_augmentations(tmp_cfg, False) + [T.Transpose()])
            transform_list = [transform] * cfg.TEST.VOTE.NUM_VOTE
        elif cfg.TEST.VOTE.TYPE == 'MULTI_VIEW':
            # build new transform
            transform_list = []
            for view_ind in range(cfg.TEST.VOTE.NUM_VOTE):
                aug_type = T.RotateByAngleWithNormal if cfg.INPUT.USE_NORMAL else T.RotateByAngle
                rotate_by_angle = aug_type(
                    cfg.TEST.VOTE.MULTI_VIEW.AXIS,
                    2 * np.pi * view_ind / cfg.TEST.VOTE.NUM_VOTE
                )
                t = [T.ToTensor(), rotate_by_angle, T.Transpose()]
                if cfg.TEST.VOTE.MULTI_VIEW.SHUFFLE:
                    # Some non-deterministic algorithms, like PointNet++, benefit from shuffle.
                    t.insert(-1, T.Shuffle())
                transform_list.append(T.Compose(t))
        else:
            raise NotImplementedError('Unsupported voting method.')

        with torch.no_grad():
            tmp_dataloader = DataLoader(test_dataset, num_workers=1, collate_fn=lambda x: x[0])
            start_time = time.time()
            end = start_time
            for ind, data in enumerate(tmp_dataloader):
                data_time = time.time() - end
                points = data['points']

                # convert points into tensor
                points_batch = [t(points.copy()) for t in transform_list]
                points_batch = torch.stack(points_batch, dim=0)
                points_batch = points_batch.cuda(non_blocking=True)

                preds = model({'points': points_batch})
                cls_logit_batch = preds['cls_logit'].cpu().numpy()  # (batch_size, num_classes)
                cls_logit_ensemble = np.mean(cls_logit_batch, axis=0)
                pred_label = np.argmax(cls_logit_ensemble)
                evaluator.update(pred_label, data['cls_label'])

                batch_time = time.time() - end
                end = time.time()

                if cfg.TEST.LOG_PERIOD > 0 and ind % cfg.TEST.LOG_PERIOD == 0:
                    logger.info('iter: {:4d}  time:{:.4f}  data:{:.4f}'.format(ind, batch_time, data_time))
        test_time = time.time() - start_time
        logger.info('Test total time: {:.2f}s'.format(test_time))
    else:
        test_meters = MetricLogger(delimiter='  ')
        test_meters.bind(metric)
        with torch.no_grad():
            start_time = time.time()
            end = start_time
            for iteration, data_batch in enumerate(test_dataloader):
                data_time = time.time() - end

                cls_label_batch = data_batch['cls_label'].numpy()
                data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

                preds = model(data_batch)

                loss_dict = loss_fn(preds, data_batch)
                total_loss = sum(loss_dict.values())

                test_meters.update(loss=total_loss, **loss_dict)
                metric.update_dict(preds, data_batch)

                cls_logit_batch = preds['cls_logit'].cpu().numpy()  # (batch_size, num_classes)
                pred_label_batch = np.argmax(cls_logit_batch, axis=1)
                evaluator.batch_update(pred_label_batch, cls_label_batch)

                batch_time = time.time() - end
                end = time.time()
                test_meters.update(time=batch_time, data=data_time)

                if cfg.TEST.LOG_PERIOD > 0 and iteration % cfg.TEST.LOG_PERIOD == 0:
                    logger.info(
                        test_meters.delimiter.join(
                            [
                                'iter: {iter:4d}',
                                '{meters}',
                            ]
                        ).format(
                            iter=iteration,
                            meters=str(test_meters),
                        )
                    )
        test_time = time.time() - start_time
        logger.info('Test {}  total time: {:.2f}s'.format(test_meters.summary_str, test_time))

    # evaluate
    logger.info('overall accuracy={:.2f}%'.format(100.0 * evaluator.overall_accuracy))
    logger.info('average class accuracy={:.2f}%.\n{}'.format(
        100.0 * np.nanmean(evaluator.class_accuracy), evaluator.print_table()))
    evaluator.save_table(osp.join(output_dir, 'eval.cls.tsv'))


def main():
    args = parse_args()

    # Load the configuration
    from shaper.config.classification import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # Replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace('configs', 'outputs')
        output_dir = output_dir.replace('@', config_path)
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('shaper', output_dir, prefix='test')
    logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
    logger.info(args)

    from core.utils.torch_util import collect_env_info
    logger.info('Collecting env info (might take some time)\n' + collect_env_info())

    logger.info('Loaded configuration file {}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.TASK == 'classification'
    test(cfg, output_dir)


if __name__ == '__main__':
    main()
