# Openlane-V2 Topology Understanding
## Ref:

https://github.com/OpenDriveLab/OpenLane-V2

https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc6

https://github.com/microsoft/DeepSpeed

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

### 6/20/2023

1.  The program now supports running with  fp16 on Ubuntu (WSL2). With a batch_size of 2 and gradient_accumulation_steps of 2, the baseline can be run on a single RTX 3080 16G graphics card. Previously, the default precision was fp32 with a batch_size of 1, which required 22G of VRAM.

    ```
    Tue Jun 20 23:04:01 2023
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 535.43.02              Driver Version: 535.98       CUDA Version: 12.2     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  NVIDIA GeForce RTX 3080 ...    On  | 00000000:01:00.0  On |                  N/A |
    | N/A   63C    P0              51W / 120W |  16088MiB / 16384MiB |    100%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
    ```


2. The program now also supports fp16 configuration with the DeepSpeed framework.

     ```python
       gradient_accumulation_steps = 2

       ds_config = {
           "train_micro_batch_size_per_gpu": cfg.data.samples_per_gpu,
           "gradient_accumulation_steps": gradient_accumulation_steps,
           "optimizer": {
               "type": "Adam",
               "params": {
                   "lr": 1e-4
               }
           },

           "fp16": {
               "enabled": True,    
      },
       }
```

log:

```
2023-06-20 23:06:28,577 - mmdet - INFO - Epoch 0, idx 349 / 11239, iter 174 / 5619, bs 2 *acc 2: 4, eta 2 days, 9:12:06, iter_time 0:00:37, loss 5.2290, log_vars : OrderedDict([('lc_loss_cls', 0.12676870822906494), ('lc_loss_bbox', 5.594433784484863), ('te_loss_cls', 1.4661476612091064), ('te_loss_bbox', 1.3309221267700195), ('te_loss_iou', 1.3860794305801392), ('topology_lclc_loss_cls', 0.276611328125), ('topology_lcte_loss_cls', 0.277099609375), ('loss', 10.458063125610352)])
```





