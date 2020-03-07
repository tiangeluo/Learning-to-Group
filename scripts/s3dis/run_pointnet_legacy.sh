#!/usr/bin/env bash
CFG="configs/s3dis_legacy/pointnet_sem_seg.yaml"
GPU=0
AREA=$1
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpu)
            GPU=$2
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

CUDA_VISIBLE_DEVICES=${GPU} python s3dis/train_pointnet_legacy.py --cfg=${CFG} DATASET.TEST ${AREA} $@
CUDA_VISIBLE_DEVICES=${GPU} python s3dis/test_pointnet_legacy.py --cfg=${CFG} DATASET.TEST ${AREA} $@