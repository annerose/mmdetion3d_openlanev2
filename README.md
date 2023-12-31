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


3. log result:

    ```
    2023-06-20 23:06:28,577 - mmdet - INFO - Epoch 0, idx 349 / 11239, iter 174 / 5619, bs 2 *acc 2: 4, eta 2 days, 9:12:06, iter_time 0:00:37, loss 5.2290, log_vars : OrderedDict([('lc_loss_cls', 0.12676870822906494), ('lc_loss_bbox', 5.594433784484863), ('te_loss_cls', 1.4661476612091064), ('te_loss_bbox', 1.3309221267700195), ('te_loss_iou', 1.3860794305801392), ('topology_lclc_loss_cls', 0.276611328125), ('topology_lcte_loss_cls', 0.277099609375), ('loss', 10.458063125610352)])
    ```



### 6/21/2023

1. Compatible with Windows CPUs and Ubuntu DeepSpeed mode, applicable parameter is --ds.
2. Add the parameter --fp-16

```shell
# Ubuntu DeepSpeed
cd  mmdetection3d-1.0.0rc6
python tools/train_ds.py projects/openlanev2/configs/baseline.py --use-ds --use-fp16

# windows
python tools/train_ds.py projects/openlanev2/configs/baseline_cpu.py
```



3. Inline parse_batch_data_container to reduce iter_time from 37s to 7s, reason unknown.

```
2023-06-21 14:56:05,144 - mmdet - INFO - Epoch 0, idx 9 / 11239, iter 4 / 5619, bs 2 *acc 2: 4, eta 11:53:28, iter_time 0:00:07, loss 7.2697, log_vars : OrderedDict([('lc_loss_cls', 0.19211412966251373), ('lc_loss_bbox', 5.803062915802002), ('te_loss_cls', 3.4926228523254395), ('te_loss_bbox', 2.9603238105773926), ('te_loss_iou', 1.0365519523620605), ('topology_lclc_loss_cls', 0.50341796875), ('topology_lcte_loss_cls', 0.55126953125), ('loss', 14.539363861083984)])
```

### 6/22/2023

1. Add gpu tflops benchmark for fp16 and fp32

   ```python
   python mytest.py 
   ```

   |                                    | VRAM(G) | fp16  | fp32  |
   | ---------------------------------- | ------- | ----- | ----- |
   | RTX 3080 Laptop (Win10)            | 16G     | 23.06 | 14.76 |
   | RTX 3080 Laptop (WSL2 Ubuntu22.04) | 16G     | 35.54 | 16.07 |
   | Tesla V100-PCIE-32GB               | 31.74GB | 85.93 | 13.60 |


### 6/29/2023

1. Support pytorch checkpoint

   Checkpointing works by trading compute for memory. Rather than storing all intermediate activations of the entire computation graph for computing backward, the checkpointed part does **not** save intermediate activations, and instead recomputes them in backward pass. It can be applied on any part of a model.

```python
#         img_feats = self.extract_feat(img=img, img_metas=img_metas)
        img_feats = cp.checkpoint(self.extract_feat,img, img_metas)

#         bev_feats = self.bev_constructor(img_feats, img_metas, prev_bev)
        bev_feats = cp.checkpoint(self.bev_constructor, img_feats, img_metas, prev_bev)

```

test large model on v100 (vram 32G)

original vram : batch = 1：  18G 

use checkpoint:

batch_size = 1:   	8G

batch_size = 2: 	17G

batch_size = 3: 	29G

2. Support lr_scheduler in  deepspeed

### 7/1/2023

1. Support dist deepspeed fp16 for large model

   does not require a hostfile for signle-node multi-gpu

   ```shell
   deepspeed    --num_gpus=4 tools/train_ds.py projects/openlanev2/configs/baseline_large_v100.py --use-ds --use-fp16 --seed 0 --launcher pytorch
   
   ```
   
   
