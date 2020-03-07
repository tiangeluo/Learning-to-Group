#!/usr/bin/env bash
CFG="configs/baselines/$1_cls.yaml"
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

CUDA_VISIBLE_DEVICES=${GPU} python shaper/train_cls.py --cfg=${CFG} $@
CUDA_VISIBLE_DEVICES=${GPU} python shaper/test_cls.py --cfg=${CFG} $@