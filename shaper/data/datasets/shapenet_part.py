"""ShapeNetCore Part Segmentation

For PointNet, the authors have sampled 2048 points from ShapeNetCore to generate HDF5.
Notice that their released codes use data only verified by experts, which is same as shapenetcore_benchmark_v0.

References:
    http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html

"""

import os.path as osp
from collections import OrderedDict

import h5py
import json
import numpy as np

from torch.utils.data import Dataset
from shaper.utils.pc_util import normalize_points


class ShapeNetPartH5(Dataset):
    """ShapeNetCore HDF5 dataset

    HDF5 data has already converted catid_partid to a global seg_id.

    Args:
        root_dir (str): the root directory of data.
        split (str): the names of dataset, e.g. ['train', 'test']
        transform (object): methods to transform inputs.
        num_points (int): the number of input points. -1 means using all.
        load_seg (bool): whether to load segmentation labels

    Attributes:
        class_names (list): the names of classes
        class_to_seg_map (dict): mapping from class labels to seg labels
        meta_data (list of dict): meta information of data

    TODO:
        Add the description of how points are sampled from raw data.

    """
    url = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
    cat_file = 'all_object_categories.txt'
    seg_file = 'overallid_to_catid_partid.json'
    split_map = {
        'train': 'train_hdf5_file_list.txt',
        'val': 'val_hdf5_file_list.txt',
        'test': 'test_hdf5_file_list.txt',
    }

    def __init__(self, root_dir, split, transform=None, num_points=-1, load_seg=False):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.transform = transform
        self.load_seg = load_seg

        # classes
        self.class_to_catid_map = self._load_cat_file()
        self.catid_to_class_map = {v: k for k, v in self.class_to_catid_map.items()}
        self.class_names = list(self.class_to_catid_map.keys())
        self.class_to_ind_map = {c: i for i, c in enumerate(self.class_names)}

        # segid to (catid, partid). Notice that partid start from 1
        if self.load_seg:
            self.segid_to_catid_partid_map = self._load_seg_file()
            self.class_to_seg_map = {}
            for cls, catid in self.class_to_catid_map.items():
                class_ind = self.class_to_ind_map[cls]
                segids = [segid for segid, x in enumerate(self.segid_to_catid_partid_map) if x[0] == catid]
                self.class_to_seg_map[class_ind] = segids

        # meta data and cache
        self.meta_data = []
        self.cache = {
            'points': [],
            'cls_label': [],
            'seg_label': [],
        }

        for split_name in self.split.split('+'):
            self._load_dataset(split_name)

        for k, v in self.cache.items():
            self.cache[k] = np.concatenate(v, axis=0)

        print('{:s}: {} classes with {} shapes'.format(
            self.__class__.__name__, len(self.class_names), len(self.meta_data)))

    def _load_cat_file(self):
        class_to_catid_map = OrderedDict()
        with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
            for line in fid:
                class_name, catid = line.strip().split()
                class_to_catid_map[class_name] = catid
        return class_to_catid_map

    def _load_seg_file(self):
        return json.load(open(osp.join(self.root_dir, self.seg_file), 'r'))

    def _load_dataset(self, split_name):
        split_fname = osp.join(self.root_dir, self.split_map[split_name])
        fname_list = [line.rstrip() for line in open(split_fname, 'r')]

        for fname in fname_list:
            data_path = osp.join(self.root_dir, osp.basename(fname))
            with h5py.File(data_path, mode='r') as f:
                num_samples = f['label'].shape[0]
                self.cache['points'].append(f['data'][:])
                self.cache['cls_label'].append(f['label'][:].squeeze(1).astype(int))
                if self.load_seg:
                    self.cache['seg_label'].append(f['pid'][:].astype(int))
            for ind in range(num_samples):
                self.meta_data.append({
                    'offset': ind,
                    'size': num_samples,
                    'path': data_path,
                })

    def __getitem__(self, index):
        points = self.cache['points'][index].astype(np.float32)
        cls_label = int(self.cache['cls_label'][index])
        seg_label = self.cache['seg_label'][index].astype(np.int64) if self.load_seg else None
        out_dict = {}

        if self.num_points > 0:
            assert self.num_points <= points.shape[0]
            # refer to original implementations of PointNet.
            points = points[:self.num_points]
            if seg_label is not None:
                seg_label = seg_label[:self.num_points]

        if self.transform is not None:
            points, seg_label = self.transform(points, seg_label)

        out_dict['points'] = points
        out_dict['cls_label'] = cls_label
        if seg_label is not None:
            out_dict['seg_label'] = seg_label

        return out_dict

    def __len__(self):
        return len(self.meta_data)


class ShapeNetPart(Dataset):
    """ShapeNetCore part segmentation dataset

    Each class of ShapeNetCore is assigned a catid/offset, like '02691156'.
    Each part is associated with one class.
    - 16 object categories (airplane, chair, motorbike)
    - 50 part classes (each object category has 2-6 part classes)

    Attributes:
        class_names (list): the names of classes
        class_to_seg_map (dict): mapping from class labels to segmentation labels
        meta_data (list of dict): meta information of data

    """
    url = 'https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip'
    cat_file = 'synsetoffset2category.txt'
    split_dir = 'train_test_split'
    segid_to_catid_partid_map = [['02691156', 1], ['02691156', 2], ['02691156', 3], ['02691156', 4], ['02773838', 1],
                                 ['02773838', 2], ['02954340', 1], ['02954340', 2], ['02958343', 1], ['02958343', 2],
                                 ['02958343', 3], ['02958343', 4], ['03001627', 1], ['03001627', 2], ['03001627', 3],
                                 ['03001627', 4], ['03261776', 1], ['03261776', 2], ['03261776', 3], ['03467517', 1],
                                 ['03467517', 2], ['03467517', 3], ['03624134', 1], ['03624134', 2], ['03636649', 1],
                                 ['03636649', 2], ['03636649', 3], ['03636649', 4], ['03642806', 1], ['03642806', 2],
                                 ['03790512', 1], ['03790512', 2], ['03790512', 3], ['03790512', 4], ['03790512', 5],
                                 ['03790512', 6], ['03797390', 1], ['03797390', 2], ['03948459', 1], ['03948459', 2],
                                 ['03948459', 3], ['04099429', 1], ['04099429', 2], ['04099429', 3], ['04225987', 1],
                                 ['04225987', 2], ['04225987', 3], ['04379243', 1], ['04379243', 2], ['04379243', 3]]

    split_map = {
        'train': 'shuffled_train_file_list.json',
        'val': 'shuffled_val_file_list.json',
        'test': 'shuffled_test_file_list.json',
    }

    def __init__(self, root_dir, split, transform=None, num_points=-1,
                 normalize=True, load_seg=False):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.normalize = normalize
        self.transform = transform
        self.load_seg = load_seg

        # classes
        self.class_to_catid_map = self._load_cat_file()
        self.catid_to_class_map = {v: k for k, v in self.class_to_catid_map.items()}
        self.class_names = list(self.class_to_catid_map.keys())
        self.class_to_ind_map = {c: i for i, c in enumerate(self.class_names)}

        # segid to (catid, partid). Notice that partid start from 1
        if self.load_seg:
            self.class_to_seg_map = {}
            for cls, catid in self.class_to_catid_map.items():
                class_ind = self.class_to_ind_map[cls]
                segids = [segid for segid, x in enumerate(self.segid_to_catid_partid_map) if x[0] == catid]
                self.class_to_seg_map[class_ind] = segids

        # meta data
        self.meta_data = []
        for split_name in self.split.split('+'):
            meta_data = self._load_dataset(split_name)
            self.meta_data.extend(meta_data)

        print('{:s}: {} classes with {} shapes'.format(
            self.__class__.__name__, len(self.class_names), len(self.meta_data)))

    def _load_cat_file(self):
        class_to_catid_map = OrderedDict()
        with open(osp.join(self.root_dir, self.cat_file), 'r') as fid:
            for line in fid:
                class_name, catid = line.strip().split()
                class_to_catid_map[class_name] = catid
        return class_to_catid_map

    def _load_pts(self, fname):
        return np.loadtxt(fname).astype(np.float32)

    def _load_seg(self, fname):
        return np.loadtxt(fname).astype(int)

    def _convert_part_to_seg(self, cls_label, part_label):
        seg_label = part_label.copy()
        for segid in self.class_to_seg_map[cls_label]:
            catid, partid = self.segid_to_catid_partid_map[segid]
            seg_label[seg_label == partid] = segid
        return seg_label

    def _load_dataset(self, split_name):
        split_fname = osp.join(self.root_dir, self.split_dir, self.split_map[split_name])
        fname_list = json.load(open(split_fname, 'r'))
        meta_data = []
        for fname in fname_list:
            _, catid, token = fname.split('/')
            pts_path = osp.join(self.root_dir, catid, 'points', token + '.pts')
            class_name = self.catid_to_class_map[catid]
            data = {
                'token': token,
                'class': class_name,
                'cls_label': self.class_to_ind_map[class_name],
                'pts_path': pts_path,
            }
            if self.load_seg:
                seg_path = osp.join(self.root_dir, catid, 'points_label', token + '.seg')
                data['seg_path'] = seg_path
            meta_data.append(data)
        return meta_data

    def __getitem__(self, index):
        meta_data = self.meta_data[index]
        points = self._load_pts(meta_data['pts_path'])
        cls_label = meta_data['cls_label']
        seg_label = None
        out_dict = {}

        if self.num_points != -1:
            points = points[:self.num_points]

        if self.normalize:
            points = normalize_points(points)

        if self.load_seg:
            part_label = self._load_seg(meta_data['seg_path'])
            seg_label = self._convert_part_to_seg(cls_label, part_label)

        if self.transform is not None:
            points, seg_label = self.transform(points, seg_label)

        out_dict['points'] = points
        out_dict['cls_label'] = cls_label
        if self.load_seg:
            out_dict['seg_label'] = seg_label

        return out_dict

    def __len__(self):
        return len(self.meta_data)


class ShapeNetPartNormal(ShapeNetPart):
    """ShapeNetCore dataset, points include normal."""
    url = 'https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip'

    def __init__(self, *args, with_normal=True, **kwargs):
        super(ShapeNetPartNormal, self).__init__(*args, **kwargs)
        self.with_normal = with_normal
        self.cache = None
        self.load_cache()

    def _load_dataset(self, dataset_name):
        split_fname = osp.join(self.root_dir, self.split_dir, self.split_map[dataset_name])
        fname_list = json.load(open(split_fname, 'r'))
        meta_data = []
        for fname in fname_list:
            _, catid, token = fname.split('/')
            pts_path = osp.join(self.root_dir, catid, token + '.txt')
            class_name = self.catid_to_class_map[catid]
            data = {
                'token': token,
                'class': class_name,
                'cls_label': self.class_to_ind_map[class_name],
                'pts_path': pts_path,
            }
            meta_data.append(data)
        return meta_data

    def load_cache(self):
        import pickle
        cache_filename = osp.join(self.root_dir, 'cache.{}.pkl'.format(self.split))
        if osp.exists(cache_filename):
            print('Loading cache from', cache_filename)
            self.cache = {}
            with open(cache_filename, 'rb') as f:
                self.cache = pickle.load(f)
        else:
            print('Saving cache to', cache_filename)
            self.cache = self.generate_cache()
            with open(cache_filename, 'wb') as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    def generate_cache(self):
        from tqdm import tqdm
        from collections import defaultdict
        cache = defaultdict(list)
        for index in tqdm(range(len(self))):
            meta_data = self.meta_data[index]
            points = self._load_pts(meta_data['pts_path'])
            cache['points'].append(points)
            cache['cls_label'].append(meta_data['cls_label'])
        return dict(cache)

    def __getitem__(self, index):
        if self.cache:
            points = self.cache['points'][index].astype(np.float32)
            cls_label = int(self.cache['cls_label'][index])
        else:
            meta_data = self.meta_data[index]
            points = self._load_pts(meta_data['pts_path'])
            cls_label = meta_data['cls_label']

        seg_label = None
        out_dict = {}

        if self.num_points != -1:
            points = points[:self.num_points]

        if self.normalize:
            points[:, 0:3] = normalize_points(points[:, 0:3])

        if self.load_seg:
            seg_label = points[:, -1].astype(int)

        # discard the segmentation labels
        points = points[:, 0:6]

        if not self.with_normal:
            points = points[:, 0:3]

        if self.transform is not None:
            points, seg_label = self.transform(points, seg_label)

        out_dict['points'] = points
        out_dict['cls_label'] = cls_label
        if self.load_seg:
            out_dict['seg_label'] = seg_label

        return out_dict


def test_ShapeNetPartH5():
    import matplotlib.pyplot as plt
    from shaper.utils.visualize import visualize_point_cloud
    dataset = ShapeNetPartH5('../../../data/shapenet_part_hdf5', 'val', load_seg=True)
    data = dataset[0]
    points = data['points']
    cls_label = data['cls_label']
    seg_label = data['seg_label']
    print(points.shape, points.dtype)
    print(cls_label, dataset.class_names[cls_label])
    print(seg_label.shape, seg_label.dtype)
    # open3d.draw_geometries([draw_point_cloud(points)])

    num_parts = len(dataset.class_to_seg_map[cls_label])
    part_label = np.asarray([dataset.segid_to_catid_partid_map[label][1] for label in seg_label]) - 1
    cmap = plt.get_cmap('hsv', num_parts)
    cmap = np.asarray([cmap(i) for i in range(num_parts)])[:, :3]
    visualize_point_cloud(points, colors=cmap[part_label])


def test_ShapeNetPart():
    import matplotlib.pyplot as plt
    from shaper.utils.visualize import visualize_point_cloud
    dataset = ShapeNetPart('../../../data/shapenet_part', 'test', load_seg=True)
    data = dataset[0]
    points = data['points']
    cls_label = data['cls_label']
    seg_label = data['seg_label']
    print(points.shape, points.dtype)
    print(cls_label, dataset.class_names[cls_label])
    print(seg_label.shape, seg_label.dtype)
    # open3d.draw_geometries([draw_point_cloud(points)])

    num_parts = len(dataset.class_to_seg_map[cls_label])
    part_label = np.asarray([dataset.segid_to_catid_partid_map[label][1] for label in seg_label]) - 1
    cmap = plt.get_cmap('hsv', num_parts)
    cmap = np.asarray([cmap(i) for i in range(num_parts)])[:, :3]
    visualize_point_cloud(points, colors=cmap[part_label])


def test_ShapeNetPartNormal():
    import matplotlib.pyplot as plt
    from shaper.utils.visualize import visualize_point_cloud
    dataset = ShapeNetPartNormal('../../../data/shapenet_part_normal', 'test', load_seg=True)
    data = dataset[1]
    points = data['points']
    cls_label = data['cls_label']
    seg_label = data['seg_label']
    print(points.shape, points.dtype)
    print(cls_label, dataset.class_names[cls_label])
    print(seg_label.shape, seg_label.dtype)
    # visualize_point_cloud(points[:, 0:3])

    num_parts = len(dataset.class_to_seg_map[cls_label])
    part_label = np.asarray([dataset.segid_to_catid_partid_map[label][1] for label in seg_label]) - 1
    cmap = plt.get_cmap('hsv', num_parts)
    cmap = np.asarray([cmap(i) for i in range(num_parts)])[:, :3]
    visualize_point_cloud(points[:, 0:3], colors=cmap[part_label], normals=points[:, 3:6])
