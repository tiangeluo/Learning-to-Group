from __future__ import division
import warnings

from torch.utils.data.dataloader import DataLoader

# from core.utils.torch_util import worker_init_fn
from shaper.data import datasets as D
from shaper.data import transforms as T


def build_dataloader(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']
    batch_size = cfg.TRAIN.BATCH_SIZE if mode == 'train' else cfg.TEST.BATCH_SIZE
    is_train = (mode == 'train')

    if cfg.TASK == 'classification':
        dataset = build_cls_dataset(cfg, mode)
    elif cfg.TASK == 'part_segmentation':
        dataset = build_part_seg_dataset(cfg, mode)
    else:
        raise NotImplementedError('Unsupported task: {}'.format(cfg.TASK))

    warnings.warn('Pytorch multiprocessing has issues with np.random. '
                  'Please avoid using np.random in your dataset,'
                  'otherwise use core.utils.torch_util.worker_init_fn')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=(is_train and cfg.DATALOADER.DROP_LAST),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        # worker_init_fn=worker_init_fn,
    )
    return dataloader


def build_cls_dataset(cfg, mode='train'):
    split = cfg.DATASET[mode.upper()]
    is_train = (mode == 'train')

    transform_list = parse_augmentations(cfg, is_train)
    transform_list.insert(0, T.ToTensor())
    transform_list.append(T.Transpose())
    transform = T.Compose(transform_list)

    if cfg.DATASET.TYPE == 'ModelNet40H5':
        dataset = D.ModelNet40H5(root_dir=cfg.DATASET.ROOT_DIR,
                                 split=split,
                                 num_points=cfg.INPUT.NUM_POINTS,
                                 transform=transform)
    elif cfg.DATASET.TYPE == 'ModelNet40':
        dataset = D.ModelNet40(root_dir=cfg.DATASET.ROOT_DIR,
                               split=split,
                               num_points=cfg.INPUT.NUM_POINTS,
                               normalize=True,
                               with_normal=cfg.INPUT.USE_NORMAL,
                               transform=transform)
    else:
        raise ValueError('Unsupported type of dataset: {}.'.format(cfg.DATASET.TYPE))

    return dataset


def parse_augmentations(cfg, is_train=True):
    transform_list = []
    augmentations = cfg.TRAIN.AUGMENTATION if is_train else cfg.TEST.AUGMENTATION
    for aug in augmentations:
        if isinstance(aug, (list, tuple)):
            method = aug[0]
            args = aug[1:]
        else:
            method = aug
            args = []
        if cfg.INPUT.USE_NORMAL and hasattr(T, method + 'WithNormal'):
            method = method + 'WithNormal'
        transform_list.append(getattr(T, method)(*args))
    return transform_list


def build_part_seg_dataset(cfg, mode='train'):
    split = cfg.DATASET[mode.upper()]
    is_train = (mode == 'train')

    transform_list = parse_augmentations(cfg, is_train)
    transform_list.insert(0, T.ToTensor())
    transform_list.append(T.Transpose())
    transform = T.ComposeSeg(transform_list)

    if cfg.DATASET.TYPE == 'ShapeNetPartH5':
        dataset = D.ShapeNetPartH5(root_dir=cfg.DATASET.ROOT_DIR,
                                   split=split,
                                   transform=transform,
                                   num_points=cfg.INPUT.NUM_POINTS,
                                   load_seg=True)
    elif cfg.DATASET.TYPE == 'ShapeNetPart':
        dataset = D.ShapeNetPart(root_dir=cfg.DATASET.ROOT_DIR,
                                 split=split,
                                 transform=transform,
                                 num_points=cfg.INPUT.NUM_POINTS,
                                 load_seg=True)
    elif cfg.DATASET.TYPE == 'ShapeNetPartNormal':
        assert cfg.INPUT.USE_NORMAL
        dataset = D.ShapeNetPartNormal(root_dir=cfg.DATASET.ROOT_DIR,
                                       split=split,
                                       transform=transform,
                                       num_points=cfg.INPUT.NUM_POINTS,
                                       with_normal=True,
                                       load_seg=True)
    else:
        raise ValueError('Unsupported type of dataset: {}.'.format(cfg.DATASET.TYPE))

    return dataset
