# Openlane-V2 Topology Understanding
## Ref:

https://github.com/OpenDriveLab/OpenLane-V2

https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6

## Changelog:

### 6/19/2023

1. Extract the training and validation loop from the mmdetection3d framework and place it in the tools/train_ds.py file.

2. Support debugging on Windows CPU. The runtime environment variable needs to include 
   ```shell
   CUDA_VISIBLE_DEVICES=-1.
   ```

3. Training launch parameters.

   ```shell
   cd  mmdetection3d-1.0.0rc6
   python tools/train_ds.py projects/openlanev2/configs/baseline.py
   ```

