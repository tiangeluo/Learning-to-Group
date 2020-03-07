from __future__ import division
from functools import partial


import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader, default_collate

from core.utils.torch_util import worker_init_fn
import shaper.models.pointnet2.functions as _F
from .partnet import PartNetInsSeg, PartNetRegionInsSeg
from shaper.data import transforms as T
from IPython import embed


def collate(batch, num_centroids, radius, num_neighbours,
            with_renorm, with_resample, with_shift, sample_method):
    data_batch = default_collate(batch)
    with torch.no_grad():
        xyz = data_batch.get('points').cuda(non_blocking=True)
        # ins_id, (batch_size, length)
        ins_id = data_batch.get('ins_id').cuda(non_blocking=True)
        batch_size, length = ins_id.size()

        # sample new points
        # (batch_size, num_centroids)
        assert sample_method in ['RND', 'FPS', 'LS', 'WBS']
        if sample_method == 'RND':
            centroid_index = torch.randint(low=0, high=length, size=(batch_size, num_centroids), device=ins_id.device)
        elif sample_method == 'LS':
            linspace = torch.linspace(-1, 1, steps=int(1/radius), device=ins_id.device)
            pseudo_centroids = torch.stack(torch.meshgrid(linspace, linspace, linspace), dim=0).view(1, 3, -1)
            # pdist, [batch_size, num_centroids, length]
            pdist = (xyz.unsqueeze(2) - pseudo_centroids.unsqueeze(3)).norm(dim=1)
            # (batch_size, num_centroids)
            _, centroid_index = pdist.min(dim=2)
        elif sample_method == 'WBS':
            num_centroids *= 2
            centroid_index = torch.randint(low=0, high=length, size=(batch_size, num_centroids), device=ins_id.device)
            # (batch_size, 3, num_centroids)
            centroid_xyz = _F.gather_points(xyz, centroid_index)
            # (batch_size, num_centroids, num_neighbours)
            neighbour_index, _ = _F.ball_query(xyz, centroid_xyz, radius, num_neighbours)
            # (batch_size, 3, num_centroids, num_neighbours)
            neighbour_xyz = _F.group_points(xyz, neighbour_index)

            neighbour_centroid_index = (neighbour_xyz - centroid_xyz.unsqueeze(-1)).abs().sum(1).argmin(dim=-1)

            resample_centroid_index = neighbour_index.gather(2, neighbour_centroid_index.unsqueeze(-1)).squeeze(-1)

            # neighbour_label, (batch_size, num_centroids, num_neighbours)
            neighbour_label = ins_id.gather(1, neighbour_index.view(batch_size, num_centroids*num_neighbours))
            neighbour_label = neighbour_label.view(batch_size, num_centroids, num_neighbours)

            # centroid_label, (batch_size, num_centroids, 1)
            centroid_label = ins_id.gather(1, resample_centroid_index).view(batch_size, num_centroids, 1)
            # (batch_size, num_centroids, num_neighbours)
            neighbour_centroid_dist = (neighbour_xyz - centroid_xyz.unsqueeze(-1)).norm(1)
            neighbour_centroid_dist = neighbour_centroid_dist * (neighbour_label != centroid_label.expand_as(neighbour_label)).float() + \
                                      neighbour_centroid_dist.max() * (neighbour_label == centroid_label.expand_as(neighbour_label)).float()
            # (batch_size, num_centroids)
            neighbour_centroid_dist, _ = neighbour_centroid_dist.min(dim=-1)
            _, select_centroid_index = neighbour_centroid_dist.topk(num_centroids//2, largest=False)
            centroid_index = centroid_index.gather(1, select_centroid_index)
        else:
            centroid_index = _F.farthest_point_sample(xyz, num_centroids)
        # (batch_size, 3, num_centroids)
        centroid_xyz = _F.gather_points(xyz, centroid_index)

        # (batch_size, num_centroids, num_neighbours)
        neighbour_index, _ = _F.ball_query(xyz, centroid_xyz, radius, num_neighbours)

        neighbour_index_purity, _ = _F.ball_query(xyz, centroid_xyz, 0.03, 64)
        neighbour_xyz_purity = _F.group_points(xyz, neighbour_index_purity)
        batch_size, num_centroids, num_neighbours = neighbour_index_purity.size()
        neighbour_label_purity = ins_id.gather(1, neighbour_index_purity.view(batch_size, num_centroids*num_neighbours))
        neighbour_label_purity = neighbour_label_purity.view(batch_size, num_centroids, num_neighbours)

        # (batch_size, 3, num_centroids, num_neighbours)
        neighbour_xyz = _F.group_points(xyz, neighbour_index)

        # TODO resample centroid_xyz and centroid_index var to stand for new centroid point
        # (batch_size, num_centroids)
        if with_resample:
            neighbour_centroid_index = torch.randint_like(centroid_index, low=0, high=num_neighbours)
        else:
            neighbour_centroid_index = (neighbour_xyz - centroid_xyz.unsqueeze(-1)).abs().sum(1).argmin(dim=-1)

        resample_centroid_index = neighbour_index.gather(2, neighbour_centroid_index.unsqueeze(-1)).squeeze(-1)

        # (batch_size, 3, num_centroids)
        resample_centroid_xyz = _F.gather_points(xyz, resample_centroid_index)

        # translation normalization
        centroid_mean = torch.mean(neighbour_xyz, -1).clone()
        neighbour_xyz -= centroid_mean.unsqueeze(-1)
        neighbour_xyz_purity -= centroid_mean.unsqueeze(-1)
        #neighbour_xyz -= centroid_xyz.unsqueeze(-1)
        resample_centroid_xyz -= centroid_xyz

        if with_renorm:
            norm_factor = neighbour_xyz.norm(dim=1).max()
            neighbour_xyz /= norm_factor
            resample_centroid_xyz /= norm_factor

        if with_shift:
            shift = neighbour_xyz.new(1).normal_(mean=0.0, std=0.01)
            neighbour_xyz += shift
            resample_centroid_xyz += shift

        batch_size, num_centroids, num_neighbours = neighbour_index.size()
        # neighbour_label, (batch_size, num_centroids, num_neighbours)
        neighbour_label = ins_id.gather(1, neighbour_index.view(batch_size, num_centroids*num_neighbours))
        neighbour_label = neighbour_label.view(batch_size, num_centroids, num_neighbours)

        # centroid_label, (batch_size, num_centroids, 1)
        centroid_label = ins_id.gather(1, resample_centroid_index).view(batch_size, num_centroids, 1)
        # ins_label, (batch_size, num_centroids, num_neighbours)
        ins_label = (neighbour_label == centroid_label.expand_as(neighbour_label)).long()
        valid_mask = ins_label.new_ones(ins_label.size())
        valid_mask[neighbour_label == 0] = 0
        valid_mask[centroid_label.expand_as(neighbour_label) == 0] = 0
        ins_label_purity = (neighbour_label_purity == centroid_label.expand_as(neighbour_label_purity)).long()
        purity_mask = (torch.sum(ins_label_purity,-1).float()/64 > 0.95)
        valid_mask[purity_mask.unsqueeze(-1).expand_as(neighbour_label) == 0] = 0
        valid_center_mask = purity_mask.new_ones(purity_mask.size())
        valid_center_mask[purity_mask == 0] = 0
        centroid_valid_mask = purity_mask.new_ones(purity_mask.size())
        centroid_valid_mask[purity_mask==0] = 0
        centroid_valid_mask[centroid_label.squeeze(-1) == 0] =0

        data_batch['neighbour_xyz'] = neighbour_xyz
        data_batch['neighbour_xyz_purity'] = neighbour_xyz_purity
        data_batch['neighbour_index'] = neighbour_index
        data_batch['centroid_xyz'] = resample_centroid_xyz
        data_batch['centroid_index'] = resample_centroid_index
        data_batch['centroid_label'] = centroid_label
        data_batch['centroid_valid_mask'] = centroid_valid_mask
        data_batch['neighbour_centroid_index'] = neighbour_centroid_index
        data_batch['ins_label'] = ins_label
        data_batch['valid_mask'] = valid_mask
        data_batch['valid_center_mask'] = valid_center_mask

        return data_batch


def parse_augmentations(augmentations):
    transform_list = []
    for aug in augmentations:
        if isinstance(aug, (list, tuple)):
            method = aug[0]
            args = aug[1:]
        else:
            method = aug
            args = []
        transform_list.append(getattr(T, method)(*args))
    return transform_list


def build_dataloader(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']
    is_train = (mode == 'train')
    batch_size = cfg.TRAIN.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE

    if cfg.TASK == 'ins_seg_3d':
        dataset = build_ins_seg_3d_dataset(cfg, mode)
    else:
        raise NotImplementedError('Unsupported task: {}'.format(cfg.TASK))

    if cfg.DATASET.TYPE == 'PartNetInsSeg':
        kwargs_dict = cfg.DATALOADER.KWARGS
        collate_fn = partial(collate,
                             num_centroids=kwargs_dict.num_centroids,
                             radius=kwargs_dict.radius,
                             num_neighbours=kwargs_dict.num_neighbours,
                             with_renorm=kwargs_dict.with_renorm,
                             with_resample=kwargs_dict.with_resample if is_train else False,
                             with_shift=kwargs_dict.with_shift if is_train else False,
                             sample_method=kwargs_dict.get('sample_method', 'FPS'))
    else:
        collate_fn = default_collate
    if is_train:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=cfg.DATALOADER.DROP_LAST,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    return dataloader


def build_ins_seg_3d_dataset(cfg, mode='train'):
    is_train = (mode == 'train')

    augmentations = cfg.TRAIN.AUGMENTATION if is_train else cfg.TEST.AUGMENTATION
    transform_list = parse_augmentations(augmentations)
    transform_list.insert(0, T.ToTensor())
    transform_list.append(T.Transpose())
    transform = T.ComposeSeg(transform_list)

    kwargs_dict = cfg.DATASET[cfg.DATASET.TYPE].get(mode.upper(), dict())

    if cfg.DATASET.TYPE == 'PartNetInsSeg':
        dataset = PartNetInsSeg(root_dir=cfg.DATASET.ROOT_DIR,
                                transform=transform,
                                **kwargs_dict)
    elif cfg.DATASET.TYPE == 'PartNetRegionInsSeg':
        dataset = PartNetRegionInsSeg(root_dir=cfg.DATASET.ROOT_DIR,
                                      transform=transform,
                                      **kwargs_dict)
    else:
        raise ValueError('Unsupported type of dataset.')

    return dataset
