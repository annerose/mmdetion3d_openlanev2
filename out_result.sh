#!/usr/bin/env bash

python tools/test.py projects/openlanev2/configs/internimage-s.py work_dirs/internimage-s/epoch_8.pth --eval 'OpenLane-V2 Score' 'F-Score for 3D Lane' --eval-options dump=True dump_dir=work_dirs/internimage-s