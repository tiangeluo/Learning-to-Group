import os
import os.path as osp
from collections import OrderedDict, defaultdict
import json

import h5py
import numpy as np
import torch

from torch.utils.data import Dataset

from shaper.utils.pc_util import normalize_points as normalize_points_np
from partnet.utils.torch_pc import group_points, select_points
from partnet.utils.torch_pc import normalize_points as normalize_points_torch
from IPython import embed


class PartNetInsSeg(Dataset):
    cat_file = './shape_names.txt'

    def __init__(self,
                 root_dir,
                 split,
                 normalize=True,
                 transform=None,
                 shape='',
                 stage1='',
                 level=-1,
                 cache_mode=True
                 ):
        self.root_dir = root_dir
        self.split = split
        self.normalize = normalize
        self.transform = transform
        self.shape_levels = self._load_cat_file()
        self.cache_mode = cache_mode
        self.folder_list = self._prepare_file_list(shape, level)

        self.cache = defaultdict(list)
        self.meta_data = []
        self._load_data()
        print('{:s}: with {} shapes'.format(
            self.__class__.__name__,  len(self.meta_data)))

    def _load_cat_file(self):
        # Assume that the category file is put under root_dir.
        #with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
        with open(self.cat_file, 'r') as fid:
            shape_levels = OrderedDict()
            for line in fid:
                shape, levels = line.strip().split('\t')
                levels = tuple([int(l) for l in levels.split(',')])
                shape_levels[shape] = levels
        return shape_levels

    def _prepare_file_list(self, shape, level):
        shape = shape if len(shape) > 0 else None
        level = level if level > 0 else None
        # prepare files according to shape and level, if none, all will be loaded
        folder_list = []
        if (type(shape).__name__ == 'list') == False:
            shape_arr=list()
            shape_arr.append(shape)
        else:
            shape_arr = shape

        for shape in shape_arr:
            if shape is not None:
                if level is not None:
                    assert level in self.shape_levels[shape], '{} not in {}'.format(level, shape)
                    folder_list.append('{}-{}'.format(shape, level))
                else:
                    for l in self.shape_levels[shape]:
                        folder_list.append('{}-{}'.format(shape, l))
            else:
                if level is not None:
                    for shape, levels in self.shape_levels.items():
                        if level in levels:
                            folder_list.append('{}-{}'.format(shape, level))
                else:
                    for shape, levels in self.shape_levels.items():
                        for l in levels:
                            folder_list.append('{}-{}'.format(shape, l))
        return folder_list

    def _load_data(self):
        for folder in self.folder_list:
            folder_path = osp.join(self.root_dir, folder)
            files = os.listdir(folder_path)
            num_list = []
            for f in files: 
                num_list.append(int(f.split('-')[-1].split('.')[0]))
            rank = np.argsort(np.array(num_list))
            #for fname in os.listdir(folder_path):
            for k in rank:
                fname = files[k]
                if fname.startswith(self.split) and fname.endswith('h5'):
                    if self.split=='test':
                        folder_path = folder_path.replace('for_detection', 'gt')
                        data_path = osp.join(folder_path, fname)
                        print('loading {}'.format(data_path))
                        with h5py.File(data_path, mode='r') as f:
                            num_samples = f['pts'].shape[0]
                            if self.cache_mode:
                                # point cloud [N, 10000, 3]
                                self.cache['points'].append(f['pts'][:])
                                # instance idx [N, 200, 10000]
                                self.cache['gt_mask'].append(f['gt_mask'][:])
                                # valid class indicator [N, 200]
                                self.cache['gt_valid'].append(f['gt_mask_valid'][:])
                                # valid class indicator [N, 10000]
                                self.cache['gt_other_mask'].append(f['gt_mask_other'][:])
                                gt_valid = f['gt_mask_valid'][:]
                                gt_mask = f['gt_mask'][:]
                                gt_label = f['gt_mask_label'][:]
                                real_gt_label = np.zeros([gt_label.shape[0], 10000]).astype(np.uint8)
                                for i in range(real_gt_label.shape[0]):
                                    for j in range(np.sum(gt_valid[i])):
                                        real_gt_label[i][gt_mask[i,j]] = gt_label[i,j]
                                # semantics class [N, 10000]
                                self.cache['gt_label'].append(real_gt_label)
                    else:
                        data_path = osp.join(folder_path, fname)
                        print('loading {}'.format(data_path))
                        with h5py.File(data_path, mode='r') as f:
                            num_samples = f['pts'].shape[0]
                            if self.cache_mode:
                                # point cloud [N, 10000, 3]
                                self.cache['points'].append(f['pts'][:])
                                # semantics class [N, 10000]
                                self.cache['gt_label'].append(f['gt_label'][:])
                                # instance idx [N, 200, 10000]
                                self.cache['gt_mask'].append(f['gt_mask'][:])
                                # valid class indicator [N, 200]
                                self.cache['gt_valid'].append(f['gt_valid'][:])
                                # valid class indicator [N, 10000]
                                self.cache['gt_other_mask'].append(f['gt_other_mask'][:])
                    for ind in range(num_samples):
                        self.meta_data.append({
                            'offset': ind,
                            'size': num_samples,
                            'path': data_path,
                        })
                    data_path = data_path.replace('.h5', '.json')
                    print('loading {}'.format(data_path))
                    with open(data_path) as f:
                        meta_data_list = json.load(f)
                        #assert num_samples == len(meta_data_list)
                        for ind in range(num_samples):
                            self.meta_data[len(self.meta_data) - num_samples + ind].update(
                                meta_data_list[ind]
                            )

        for k, v in self.cache.items():
            self.cache[k] = np.concatenate(v, axis=0)

    def __getitem__(self, index):
        if self.cache_mode:
            points = self.cache['points'][index]
            # (10000, )
            gt_label = self.cache['gt_label'][index]
            # (200, 10000)
            gt_mask = self.cache['gt_mask'][index]
            # (200,)
            gt_valid = self.cache['gt_valid'][index]
            # (10000, )
            gt_other_mask = self.cache['gt_other_mask'][index]
        else:
            with h5py.File(self.meta_data[index]['path'], mode='r') as f:
                ind = self.meta_data[index]['offset']
                points = f['pts'][ind]
                gt_label = f['gt_label'][ind]
                gt_mask = f['gt_mask'][ind]
                gt_valid = f['gt_valid'][ind]
                gt_other_mask = f['gt_other_mask'][ind]

        # 0 for ignore
        gt_all_mask = np.concatenate([gt_other_mask[None, :], gt_mask], axis=0)
        ins_id = gt_all_mask.argmax(axis=0)

        if self.normalize:
            points = normalize_points_np(points)
        if self.transform is not None:
            points, ins_id = self.transform(points, ins_id)

        out_dict = dict(
            points=points,
            ins_id=ins_id,
            gt_mask=np.array(gt_mask,dtype=np.uint8),
            gt_valid=np.array(gt_valid,dtype=np.uint8),
            # meta=self.meta_data[index]
            # gt_label=gt_label,
            # gt_other_mask=gt_other_mask
        )

        return out_dict

    def get(self, anno_id):
        assert isinstance(anno_id, str)
        index = -1
        for idx, meta_data in enumerate(self.meta_data):
            if anno_id == meta_data['anno_id']:
                index = idx
                break
        assert index > 0, '{} not found'.format(anno_id)
        return self[index]

    def __len__(self):
        return len(self.meta_data)


class PartNetRegionInsSeg(PartNetInsSeg):

    def __init__(self,
                 num_centroids,
                 radius,
                 num_neighbours,
                 with_renorm=True,
                 with_resample=False,
                 with_shift=False,
                 **kwargs):
        self.num_centroids = num_centroids
        self.radius = radius
        self.num_neighbours = num_neighbours
        self.with_renorm = with_renorm
        self.with_resample = with_resample
        self.with_shift = with_shift
        super(PartNetRegionInsSeg, self).__init__(**kwargs)

    def _batch_transform(self, points, seg_labels):
        batch_size = points.size(0)
        assert batch_size == seg_labels.size(0)
        points_list = []
        seg_labels_list = []
        for i in range(batch_size):
            p, s = self.transform(points[i], seg_labels[i])
            points_list.append(p)
            seg_labels_list.append(s)

        return torch.stack(points_list), torch.stack(seg_labels_list)

    def _gen_region(self, points, gt_mask, gt_other_mask):
        data_dict = dict()
        # 0 for ignore
        gt_all_mask = torch.cat([gt_other_mask.unsqueeze(1), gt_mask], dim=1)
        # [batch_size, length]
        ins_id = gt_all_mask.argmax(dim=1)

        batch_size, length = ins_id.size()

        if self.transform is not None:
            # transpose inside
            points, ins_id = self._batch_transform(points, ins_id)

        # points, [batch_size, 3, length]
        # ins_id, [batch_size, length]

        # centroid_index, [batch_size, num_centroids]
        centroid_index = torch.randint(low=0, high=length, size=(batch_size, self.num_centroids))
        # centroid, [batch_size, 3, num_centroids]
        centroid = select_points(points, centroid_index)
        # TODO ball query
        neighbour_index_list = []
        for i in range(batch_size):
            # pdist, [batch_size, num_centroids, length]
            pdist = (points[i:i+1].unsqueeze(2) - centroid[i:i+1].unsqueeze(3)).norm(dim=1)
            # (batch_size, num_centroids, num_neighbours)
            neighbour_dist, neighbour_index = pdist.topk(self.num_neighbours, largest=False)
            neighbour_index_i = (neighbour_dist < self.radius).long() * neighbour_index + \
                                (neighbour_dist > self.radius).long() * centroid_index[i:i+1].unsqueeze(2)
            neighbour_index_list.append(neighbour_index_i)
        neighbour_index = torch.cat(neighbour_index_list)
        #
        # # pdist, [batch_size, num_centroids, length]
        # pdist = (points.unsqueeze(2) - centroid.unsqueeze(3)).norm(dim=1)
        # # (batch_size, num_centroids, num_neighbours)
        # neighbour_dist, neighbour_index = pdist.topk(self.num_neighbours, largest=False)
        # neighbour_index = (neighbour_dist < self.radius).long() * neighbour_index + \
        #                   (neighbour_dist > self.radius).long() * centroid_index.unsqueeze(2)
        neighbour = group_points(points, neighbour_index)

        # [batch_size, num_centroids]
        if self.with_resample:
            neighbour_centroid_index = torch.randint_like(centroid_index, low=0, high=batch_size)
        else:
            neighbour_centroid_index = (neighbour - centroid.unsqueeze(-1)).abs().sum(1).argmin(dim=-1)

        resample_centroid_index = neighbour_index.gather(2, neighbour_centroid_index.unsqueeze(-1)).squeeze(-1)

        # [batch_size, 3, num_centroids]
        resample_centroid = select_points(points, resample_centroid_index)

        # translation normalization
        neighbour -= centroid.unsqueeze(-1)
        resample_centroid -= centroid

        if self.with_renorm:
            norm_factor = neighbour.norm(dim=1).max()
            neighbour /= norm_factor
            resample_centroid /= norm_factor

        if self.with_shift:
            shift = neighbour.new(1).normal_(mean=0.0, std=0.01)
            neighbour += shift
            resample_centroid += shift

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

        data_dict['points'] = points
        data_dict['ins_id'] = ins_id
        data_dict['neighbour_xyz'] = neighbour
        data_dict['neighbour_index'] = neighbour_index
        data_dict['centroid_xyz'] = resample_centroid
        data_dict['centroid_index'] = resample_centroid_index
        data_dict['neighbour_centroid_index'] = neighbour_centroid_index
        data_dict['ins_label'] = ins_label
        data_dict['valid_mask'] = valid_mask
        return data_dict

    def _load_data(self):
        for folder in self.folder_list:
            folder_path = osp.join(self.root_dir, folder)
            for fname in os.listdir(folder_path):
                if fname.startswith(self.split) and fname.endswith('h5'):
                    data_path = osp.join(folder_path, fname)
                    print('loading {}'.format(data_path))
                    with h5py.File(data_path, mode='r') as f:
                        num_samples = f['pts'].shape[0]
                        # point cloud [N, 10000, 3]
                        points = torch.from_numpy(f['pts'][:])
                        if self.normalize:
                            points = normalize_points_torch(points)
                        # instance idx [N, 200, 10000]
                        gt_mask = torch.from_numpy(f['gt_mask'][:].astype(np.uint8))
                        # valid class indicator [N, 10000]
                        gt_other_mask = torch.from_numpy(f['gt_other_mask'][:].astype(np.uint8))

                    data_dict = self._gen_region(points, gt_mask, gt_other_mask)
                    for k, v in data_dict.items():
                        self.cache[k].append(v)

                    for ind in range(num_samples):
                        self.meta_data.append({
                            'offset': ind,
                            'size': num_samples,
                            'path': data_path,
                        })
                    data_path = data_path.replace('.h5', '.json')
                    print('loading {}'.format(data_path))
                    with open(data_path) as f:
                        meta_data_list = json.load(f)
                        assert num_samples == len(meta_data_list)
                        for ind in range(num_samples):
                            self.meta_data[len(self.meta_data) - num_samples + ind].update(
                                meta_data_list[ind]
                            )

        for k, v in self.cache.items():
            self.cache[k] = torch.cat(v, dim=0)

    def __getitem__(self, index):
        out_dict = dict()
        for k, v in self.cache.items():
            out_dict[k] = v[index]

        return out_dict


