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
    def __init__(self, class_names, class_to_seg_map):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.class_to_seg_map = class_to_seg_map

        self.seg_acc_per_class = [0.0 for _ in range(self.num_classes)]
        self.num_inst_per_class = [0 for _ in range(self.num_classes)]
        self.iou_per_class = [0.0 for _ in range(self.num_classes)]

    def update(self, pred_seg_logit, gt_cls_label, gt_seg_label):
        """Update per instance

        Args:
            pred_seg_logit (np.ndarray): (num_seg_classes, num_points1)
            gt_cls_label (int):
            gt_seg_label (np.ndarray): (num_points2,)

        """
        gt_cls_label = int(gt_cls_label)
        assert 0 <= gt_cls_label < self.num_classes
        segids = self.class_to_seg_map[gt_cls_label]
        num_valid_points = min(pred_seg_logit.shape[1], gt_seg_label.shape[0])
        pred_seg_logit = pred_seg_logit[segids, :num_valid_points]
        # pred_seg_logit = pred_seg_logit[:, :num_valid_points]
        gt_seg_label = gt_seg_label[:num_valid_points]

        pred_seg_label = np.argmax(pred_seg_logit, axis=0)
        for ind, segid in enumerate(segids):
            # convert partid to segid
            pred_seg_label[pred_seg_label == ind] = segid

        tp_mask = (pred_seg_label == gt_seg_label)
        seg_acc = np.mean(tp_mask)
        self.seg_acc_per_class[gt_cls_label] += seg_acc
        self.num_inst_per_class[gt_cls_label] += 1

        iou_per_instance = 0.0
        for ind, segid in enumerate(segids):
            gt_mask = (gt_seg_label == segid)
            num_intersection = np.sum(np.logical_and(tp_mask, gt_mask))
            num_pos = np.sum(pred_seg_label == segid)
            num_gt = np.sum(gt_mask)
            num_union = num_pos + num_gt - num_intersection
            iou = num_intersection / num_union if num_union > 0 else 1.0
            iou_per_instance += iou
        iou_per_instance /= len(segids)
        self.iou_per_class[gt_cls_label] += iou_per_instance

    def batch_update(self, pred_seg_logits, gt_cls_labels, gt_seg_labels):
        assert len(pred_seg_logits) == len(gt_cls_labels) == len(gt_seg_labels)
        for pred_seg_logit, gt_cls_label, gt_seg_label in zip(pred_seg_logits, gt_cls_labels, gt_seg_labels):
            self.update(pred_seg_logit, gt_cls_label, gt_seg_label)

    @property
    def overall_seg_acc(self):
        return sum(self.seg_acc_per_class) / sum(self.num_inst_per_class)

    @property
    def overall_iou(self):
        return sum(self.iou_per_class) / sum(self.num_inst_per_class)

    @property
    def class_seg_acc(self):
        return [seg_acc / num_inst if num_inst > 0 else float('nan')
                for seg_acc, num_inst in zip(self.seg_acc_per_class, self.num_inst_per_class)]

    @property
    def class_iou(self):
        return [iou / num_inst if num_inst > 0 else float('nan')
                for iou, num_inst in zip(self.iou_per_class, self.num_inst_per_class)]

    def print_table(self):
        from tabulate import tabulate
        header = ['Class', 'SegAccuracy', 'IOU', 'Total']
        table = []
        seg_acc_per_class = self.class_seg_acc
        iou_per_class = self.class_iou
        for ind, class_name in enumerate(self.class_names):
            table.append([class_name,
                          100.0 * seg_acc_per_class[ind],
                          100.0 * iou_per_class[ind],
                          self.num_inst_per_class[ind]
                          ])
        return tabulate(table, headers=header, tablefmt='psql', floatfmt='.2f')

    def save_table(self, filename):
        from tabulate import tabulate
        header = ['overall acc', 'overall iou'] + self.class_names
        table = [[self.overall_seg_acc, self.overall_iou] + self.class_iou]
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
    evaluator = Evaluator(test_dataset.class_names, test_dataset.class_to_seg_map)

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
                cls_label = data['cls_label']
                seg_label = data['seg_label']
                num_points = points.shape[0]

                # convert points into tensor
                points_batch = [t(points.copy()) for t in transform_list]
                if cfg.TEST.VOTE.SHUFFLE:
                    index_batch = [torch.randperm(num_points) for _ in points_batch]
                    points_batch = [points[index] for points, index in zip(points_batch, index_batch)]
                points_batch = torch.stack(points_batch, dim=0)
                points_batch = points_batch.cuda(non_blocking=True)
                cls_label_batch = torch.tensor([cls_label] * cfg.TEST.VOTE.NUM_VOTE).cuda(non_blocking=True)

                preds = model({'points': points_batch, 'cls_label': cls_label_batch})
                seg_logit_batch = preds['seg_logit'].cpu().numpy()  # (batch_size, num_seg_classes, num_points)

                if cfg.TEST.VOTE.SHUFFLE:
                    seg_logit_ensemble = np.zeros_like(seg_logit_batch[0])
                    for i, index in enumerate(index_batch):
                        index = index.numpy()
                        seg_logit_ensemble[:, index] += seg_logit_batch[i]
                else:
                    seg_logit_ensemble = np.mean(seg_logit_batch, axis=0)

                evaluator.update(seg_logit_ensemble, cls_label, seg_label)

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
                seg_label_batch = data_batch['seg_label'].numpy()
                data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

                preds = model(data_batch)

                loss_dict = loss_fn(preds, data_batch)
                total_loss = sum(loss_dict.values())

                test_meters.update(loss=total_loss, **loss_dict)
                metric.update_dict(preds, data_batch)

                seg_logit_batch = preds['seg_logit'].cpu().numpy()
                evaluator.batch_update(seg_logit_batch, cls_label_batch, seg_label_batch)

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
        logger.info('Test {}  test time: {:.2f}s'.format(test_meters.summary_str, test_time))

    # evaluate
    logger.info('overall segmentation accuracy={:.2f}%'.format(100.0 * evaluator.overall_seg_acc))
    logger.info('overall IOU={:.2f}'.format(100.0 * evaluator.overall_iou))
    logger.info('class-wise segmentation accuracy and IoU.\n{}'.format(evaluator.print_table()))
    evaluator.save_table(osp.join(output_dir, 'eval.part_seg.tsv'))


def main():
    args = parse_args()

    # Load the configuration
    from shaper.config.part_segmentation import cfg
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

    assert cfg.TASK == 'part_segmentation'
    test(cfg, output_dir)


if __name__ == '__main__':
    main()
