"""IO helpers

Notes:
    1. torch supports two methods, 'torch.save' and 'torch.load', which are analogy to pickle.

"""

import pickle
from hashlib import md5


def write_pkl(obj, filename):
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_md5(filename):
    hash_obj = md5()
    with open(filename, 'rb') as f:
        hash_obj.update(f.read())
    return hash_obj.hexdigest()
