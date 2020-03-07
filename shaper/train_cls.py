#!/usr/bin/env python
"""Train point cloud classification models"""

import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(__file__) + '/..')

import argparse
import logging
import time

import torch
from torch import nn

from core.config import purge_cfg
from core.solver.build import build_optimizer, build_scheduler
from core.nn.freezer import Freezer
from core.utils.checkpoint import Checkpointer
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.tensorboard_logger import TensorboardLogger
from core.utils.torch_util import set_random_seed

from shaper.models.build import build_model
from shaper.data.build import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Learning Training')
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


def train_one_epoch(model,
                    loss_fn,
                    metric,
                    dataloader,
                    optimizer,
                    max_grad_norm=0.0,
                    freezer=None,
                    log_period=-1):
    logger = logging.getLogger('shaper.train')
    meters = MetricLogger(delimiter='  ')
    # reset metrics
    metric.reset()
    meters.bind(metric)
    # set training mode
    model.train()
    if freezer is not None:
        freezer.freeze()
    loss_fn.train()
    metric.train()

    end = time.time()
    for iteration, data_batch in enumerate(dataloader):
        data_time = time.time() - end

        data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        preds = model(data_batch)

        # backward
        optimizer.zero_grad()
        loss_dict = loss_fn(preds, data_batch)
        total_loss = sum(loss_dict.values())

        # It is slightly faster to update metrics and meters before backward
        meters.update(loss=total_loss, **loss_dict)
        with torch.no_grad():
            metric.update_dict(preds, data_batch)

        total_loss.backward()
        if max_grad_norm > 0:
            # CAUTION: built-in clip_grad_norm_ clips the total norm.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        if log_period > 0 and iteration % log_period == 0:
            logger.info(
                meters.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )
    return meters


def validate(model,
             loss_fn,
             metric,
             dataloader,
             log_period=-1):
    logger = logging.getLogger('shaper.validate')
    meters = MetricLogger(delimiter='  ')
    # reset metrics
    metric.reset()
    meters.bind(metric)
    # set evaluate mode
    model.eval()
    loss_fn.eval()
    metric.eval()

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end

            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

            preds = model(data_batch)

            loss_dict = loss_fn(preds, data_batch)
            total_loss = sum(loss_dict.values())

            # update metrics and meters
            meters.update(loss=total_loss, **loss_dict)
            metric.update_dict(preds, data_batch)

            batch_time = time.time() - end
            meters.update(time=batch_time, data=data_time)
            end = time.time()

            if log_period > 0 and iteration % log_period == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            'iter: {iter:4d}',
                            '{meters}',
                        ]
                    ).format(
                        iter=iteration,
                        meters=str(meters),
                    )
                )
    return meters


def train(cfg, output_dir=''):
    logger = logging.getLogger('shaper.train')

    # build model
    set_random_seed(cfg.RNG_SEED)
    model, loss_fn, metric = build_model(cfg)
    logger.info('Build model:\n{}'.format(str(model)))
    model = nn.DataParallel(model).cuda()
    # model = model.cuda()

    # build optimizer
    optimizer = build_optimizer(cfg, model)

    # build lr scheduler
    scheduler = build_scheduler(cfg, optimizer)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                logger=logger)
    checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build freezer
    if cfg.TRAIN.FROZEN_PATTERNS:
        freezer = Freezer(model, cfg.TRAIN.FROZEN_PATTERNS)
        freezer.freeze(verbose=True)  # sanity check
    else:
        freezer = None

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader = build_dataloader(cfg, mode='train')
    val_period = cfg.TRAIN.VAL_PERIOD
    val_dataloader = build_dataloader(cfg, mode='val') if val_period > 0 else None

    # build tensorboard logger (optionally by comment)
    tensorboard_logger = TensorboardLogger(output_dir)

    # train
    max_epoch = cfg.SCHEDULER.MAX_EPOCH
    start_epoch = checkpoint_data.get('epoch', 0)
    best_metric_name = 'best_{}'.format(cfg.TRAIN.VAL_METRIC)
    best_metric = checkpoint_data.get(best_metric_name, None)
    logger.info('Start training from epoch {}'.format(start_epoch))
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch + 1
        scheduler.step()
        start_time = time.time()
        train_meters = train_one_epoch(model,
                                       loss_fn,
                                       metric,
                                       train_dataloader,
                                       optimizer=optimizer,
                                       max_grad_norm=cfg.OPTIMIZER.MAX_GRAD_NORM,
                                       freezer=freezer,
                                       log_period=cfg.TRAIN.LOG_PERIOD,
                                       )
        epoch_time = time.time() - start_time
        logger.info('Epoch[{}]-Train {}  total_time: {:.2f}s'.format(
            cur_epoch, train_meters.summary_str, epoch_time))

        tensorboard_logger.add_scalars(train_meters.meters, cur_epoch, prefix='train')

        # checkpoint
        if (ckpt_period > 0 and cur_epoch % ckpt_period == 0) or cur_epoch == max_epoch:
            checkpoint_data['epoch'] = cur_epoch
            checkpoint_data[best_metric_name] = best_metric
            checkpointer.save('model_{:03d}'.format(cur_epoch), **checkpoint_data)

        # validate
        if val_period > 0 and (cur_epoch % val_period == 0 or cur_epoch == max_epoch):
            start_time = time.time()
            val_meters = validate(model,
                                  loss_fn,
                                  metric,
                                  val_dataloader,
                                  log_period=cfg.TEST.LOG_PERIOD,
                                  )
            epoch_time = time.time() - start_time
            logger.info('Epoch[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_epoch, val_meters.summary_str, epoch_time))

            tensorboard_logger.add_scalars(val_meters.meters, cur_epoch, prefix='val')

            # best validation
            if cfg.TRAIN.VAL_METRIC in val_meters.meters:
                cur_metric = val_meters.meters[cfg.TRAIN.VAL_METRIC].global_avg
                if best_metric is None or cur_metric > best_metric:
                    best_metric = cur_metric
                    checkpoint_data['epoch'] = cur_epoch
                    checkpoint_data[best_metric_name] = best_metric
                    checkpointer.save('model_best', **checkpoint_data)

    logger.info('Best val-{} = {}'.format(cfg.TRAIN.VAL_METRIC, best_metric))
    return model


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from shaper.config.classification import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace('configs', 'outputs')
        output_dir = output_dir.replace('@', config_path)
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('shaper', output_dir, prefix='train')
    logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
    logger.info(args)

    from core.utils.torch_util import collect_env_info
    logger.info('Collecting env info (might take some time)\n' + collect_env_info())

    logger.info('Loaded configuration file {}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.TASK == 'classification'
    train(cfg, output_dir)


if __name__ == '__main__':
    main()
