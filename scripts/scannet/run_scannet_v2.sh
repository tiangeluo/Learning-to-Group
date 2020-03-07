#!/usr/bin/env bash
CFG="configs/scannet_3d_sem_seg/v2/$1.yaml"
GPU=0
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

CUDA_VISIBLE_DEVICES=${GPU} python scannet/train_3d_sem_seg_v2.py --cfg=${CFG} $@
CUDA_VISIBLE_DEVICES=${GPU} python scannet/test_3d_sem_seg_chunks.py --cfg=${CFG} $@