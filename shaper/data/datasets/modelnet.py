import os.path as osp

import h5py
import numpy as np

from torch.utils.data import Dataset

from shaper.utils.pc_util import normalize_points


class ModelNet40H5(Dataset):
    """ModelNet HDF5 dataset
    The hdf5 file contains (data, normal, label). 'data' include 2048 points sampled from raw data.

    Args:
        root_dir (str): the root directory of data.
        split (str): the split of dataset
        transform (object): methods to transform inputs.
        num_points (int): the number of input points. -1 means using all.

    """
    url = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    cat_file = 'shape_names.txt'
    split_map = {
        'train': 'train_files.txt',
        'test': 'test_files.txt',
    }

    def __init__(self, root_dir, split, transform=None, num_points=-1):
        self.root_dir = root_dir
        self.split = split

        self.num_points = num_points
        self.transform = transform

        self.class_names = self._load_cat_file()
        self.class_to_ind_map = {c: i for i, c in enumerate(self.class_names)}

        # load meta data and cache
        self.meta_data = []
        self.cache = dict()

        self._load_dataset(split)

        for k, v in self.cache.items():
            self.cache[k] = np.concatenate(v, axis=0)

        print('{:s}: {} classes with {} shapes'.format(
            self.__class__.__name__, len(self.class_names), len(self.meta_data)))

    def _load_cat_file(self):
        # Assume that the category file is put under root_dir.
        with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
            class_names = [line.strip() for line in fid]
        return class_names

    def _load_dataset(self, split):
        split_fname = osp.join(self.root_dir, self.split_map[split])
        fname_list = [line.rstrip() for line in open(split_fname)]
        self.cache['points'] = []
        # self.cache['normals'] = []
        self.cache['label'] = []

        for fname in fname_list:
            data_path = osp.join(self.root_dir, osp.basename(fname))
            with h5py.File(data_path, mode='r') as f:
                num_samples = f['label'].shape[0]
                self.cache['points'].append(f['data'][:])
                # self.cache['normals'].append(f['normal'][:])
                self.cache['label'].append(f['label'][:].squeeze(1))
            for ind in range(num_samples):
                self.meta_data.append({
                    'offset': ind,
                    'size': num_samples,
                    'path': data_path,
                })

    def __getitem__(self, index):
        points = self.cache['points'][index].astype(np.float32)
        cls_label = int(self.cache['label'][index])

        if self.num_points > 0:
            assert self.num_points <= points.shape[0]
            # refer to original implementations of PointNet and PointNet++.
            points = points[:self.num_points]

        if self.transform is not None:
            points = self.transform(points)

        return {
            'points': points,
            'cls_label': cls_label,
        }

    def __len__(self):
        return len(self.meta_data)

    @property
    def labels(self):
        return self.cache['label']


class ModelNet40(Dataset):
    """ModelNet40 resampled dataset"""
    url = 'https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip'
    cat_file = 'modelnet40_shape_names.txt'
    split_map = {
        'train': 'modelnet40_train.txt',
        'test': 'modelnet40_test.txt',
    }

    def __init__(self, root_dir, split, transform=None, num_points=-1,
                 normalize=True, with_normal=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_points = num_points
        self.normalize = normalize
        self.with_normal = with_normal

        # classes
        self.class_names = self._load_cat_file()
        self.class_to_ind_map = {c: i for i, c in enumerate(self.class_names)}

        # meta data
        self.meta_data = self._load_dataset(split)

        print('{:s}: {} classes with {} shapes'.format(
            self.__class__.__name__, len(self.class_names), len(self.meta_data)))

    def _load_cat_file(self):
        with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
            class_names = [line.rstrip() for line in fid]
        return class_names

    def _load_pts(self, fname):
        return np.loadtxt(fname, delimiter=',').astype(np.float32)

    def _load_dataset(self, split_name):
        meta_data = []
        with open(osp.join(self.root_dir, self.split_map[split_name]), 'r') as f:
            for line in f:
                line = line.rstrip()
                class_name = '_'.join(line.split('_')[0:-1])
                data = {
                    'class': class_name,
                    'cls_label': self.class_to_ind_map[class_name],
                    'pts_path': osp.join(self.root_dir, class_name, line + '.txt'),
                }
                meta_data.append(data)
        return meta_data

    def __getitem__(self, index):
        meta_data = self.meta_data[index]
        points = self._load_pts(meta_data['pts_path'])
        cls_label = int(meta_data['cls_label'])

        if not self.with_normal:
            points = points[:, 0:3]

        points = points[0:self.num_points, :]

        if self.normalize:
            points[:, 0:3] = normalize_points(points[:, 0:3])

        if self.transform is not None:
            points = self.transform(points)

        return {
            'points': points,
            'cls_label': cls_label,
        }

    def __len__(self):
        return len(self.meta_data)


def test_ModelNet40H5():
    from shaper.utils.visualize import visualize_point_cloud
    dataset = ModelNet40H5('../../../data/modelnet40_ply_hdf5_2048', 'test')
    idx = 0
    data = dataset[idx]
    print(idx, dataset.class_names[data['cls_label']])
    visualize_point_cloud(data['points'])


def test_ModelNet40():
    from shaper.utils.visualize import visualize_point_cloud
    dataset = ModelNet40('../../../data/modelnet40_normal_resampled', 'test')
    idx = 0
    data = dataset[idx]
    print(idx, dataset.class_names[data['cls_label']])
    visualize_point_cloud(data['points'])
