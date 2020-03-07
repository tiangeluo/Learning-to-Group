#!/usr/bin/env bash
# For PointNet++
CFG="configs/baselines/$1_part_seg.yaml"
GPU=0
LEGACY=0
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gpu)
            GPU=$2
            shift 2
            ;;
        -l|--legacy)
            LEGACY=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

CUDA_VISIBLE_DEVICES=${GPU} python shaper/train_part_seg.py --cfg=${CFG} $@
if [[ ${LEGACY} -eq 1 ]]; then
    echo "Legacy mode"
    CUDA_VISIBLE_DEVICES=${GPU} python shaper/test_part_seg.py --cfg=${CFG} $@
else
    CUDA_VISIBLE_DEVICES=${GPU} python shaper/test_part_seg.py --cfg=${CFG} \
        TEST.BATCH_SIZE 1 \
        TEST.AUGMENTATION "()" \
        $@
fi