#!/usr/bin/env bash
#   --num_gpus=4
deepspeed tools/train_ds.py projects/openlanev2/configs/baseline_large_v100.py --use-ds --use-fp16 --seed 0 --launcher pytorch

