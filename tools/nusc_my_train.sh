#!/usr/bin/env bash
CONFIG=projects/configs/$1.py
# echo $CONFIG
GPUS=$2
PORT=${PORT:-59230}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --work-dir work_dirs/$1 \
    --launcher pytorch ${@:3} \
    --deterministic \
    --cfg-options evaluation.jsonfile_prefix=work_dirs/$1/eval/results 
    # evaluation.classwise=True
