import os
import random
import numpy as np

import torch
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.collect_env import run, run_and_read_all


def get_root_dir():
    return os.path.abspath(os.path.dirname(__file__) + '/../..')


def get_PIL_version():
    try:
        import PIL
    except ImportError as e:
        return '\n No Pillow is found.'
    else:
        return '\nPillow ({})'.format(PIL.__version__)


def git_available():
    try:
        run('git version')
        return True
    except:
        return False


def get_git_rev(first=8):
    return run_and_read_all(run, 'git rev-parse HEAD')[:first]


def get_git_modifed(git_dir=None):
    git_dir = git_dir or get_root_dir()
    return run_and_read_all(run, 'git ls-files {:s} -m'.format(git_dir))


def get_git_untracked(git_dir=None):
    git_dir = git_dir or get_root_dir()
    return run_and_read_all(run, 'git ls-files {:s} --exclude-standard --others'.format(git_dir))


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += get_PIL_version()
    if git_available():
        env_str += '\nGit revision number: {:s}'.format(get_git_rev())
        env_str += '\nGit Modified\n{:s}'.format(get_git_modifed())
        # env_str += '\nGit Untrakced\n {:s}'.format(get_git_untracked())
    return env_str


def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """
    base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)
