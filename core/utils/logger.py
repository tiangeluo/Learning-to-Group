# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Jiayuan Gu
import logging
import os
import sys
import time
import socket


def setup_logger(name, save_dir, prefix='', timestamp=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        filename = 'log'
        if prefix:
            filename += '.' + prefix
        if timestamp:
            current_time = time.strftime('%m-%d_%H-%M-%S')
            filename += '.' + current_time + '.' + socket.gethostname()
        log_file = os.path.join(save_dir, filename + '.txt')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
