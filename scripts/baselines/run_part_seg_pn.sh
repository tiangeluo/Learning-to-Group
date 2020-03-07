#!/usr/bin/env bash
# For PointNet and DGCNN
CFG="configs/baselines/$1_part_seg.yaml"
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

CUDA_VISIBLE_DEVICES=${GPU} python shaper/train_part_seg.py --cfg=${CFG} $@
CUDA_VISIBLE_DEVICES=${GPU} python shaper/test_part_seg.py --cfg=${CFG} \
    INPUT.NUM_POINTS "-1" \
    TEST.BATCH_SIZE 1 \
    TEST.LOG_PERIOD 50 \
    DATASET.ROOT_DIR "data/shapenet_part" \
    DATASET.TYPE "ShapeNetPart" \
    $@
