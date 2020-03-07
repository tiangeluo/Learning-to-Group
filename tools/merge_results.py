#!/usr/bin/env python
import argparse
import sys
import os.path as osp
import glob
import csv


def read_tsv(filename):
    """Read a tsv file

    Args:
        filename (str): target tsv file

    Returns:
        tuple of OrderedDict: multiple rows in tsv

    """
    assert filename.endswith('.tsv'), 'Only accept tsv files.'
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        # Each row in reader is an OrderedDict in python3.6.
        # There may be multiple rows in a file.
        rows = tuple(reader)
    return rows


def collect_tsv(root_dir, pattern, sort=True):
    tsv_filenames = glob.glob(osp.join(root_dir, '**', pattern), recursive=True)
    if sort:
        try:
            from natsort import natsorted
            tsv_filenames = natsorted(tsv_filenames)
        except ImportError as e:
            tsv_filenames = sorted(tsv_filenames)
    result_collection = []
    for filename in tsv_filenames:
        dirname = osp.dirname(filename)
        rows = read_tsv(filename)
        for row in rows:
            row['dir'] = dirname
            # Move to the first entry
            row.move_to_end('dir', last=False)
        result_collection.extend(rows)
    return result_collection


def parse_args():
    parser = argparse.ArgumentParser(description='Merge results in form of tsv')
    parser.add_argument('-d', '--root-dir', required=True, type=str,
                        help='Root directory to merge')
    parser.add_argument('-p', '--pattern', default='eval.tsv', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    result_collection = collect_tsv(args.root_dir, args.pattern)
    fieldnames = result_collection[0].keys()
    # Output merged result. Pipe can be used to redirect to an output file.
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    # writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(result_collection)


if __name__ == '__main__':
    main()
