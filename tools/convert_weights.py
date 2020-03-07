#!/usr/bin/env python
import argparse
import os.path as osp
import shutil

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Convert weight")
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="path to weights",
        type=str,
    )
    parser.add_argument(
        "--old",
        help="old name",
        type=str,
    )
    parser.add_argument(
        "--new",
        help="new name",
        type=str,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    filename, ext = osp.splitext(args.file)
    shutil.copy(args.file, filename + '_bk' + ext)

    checkpoint_data = torch.load(args.file)
    model_state_dict = checkpoint_data['model']

    rename_keys = []
    # rename_keys = list(model_state_dict.keys())
    for key in model_state_dict.keys():
        if args.old in key:
            rename_keys.append(key)
    for key in rename_keys:
        new_name = key.replace(args.old, args.new)
        # new_name = 'module.' + key
        print("convert", key, 'to', new_name)
        model_state_dict[new_name] = model_state_dict.pop(key)
    checkpoint_data['model'] = model_state_dict
    torch.save(checkpoint_data, args.file)


if __name__ == "__main__":
    main()
