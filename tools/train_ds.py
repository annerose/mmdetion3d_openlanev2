# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import argparse
import copy
import os
import time
import warnings
from os import path as osp

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, build_optimizer

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version
from mmdet.datasets import build_dataloader as build_mmdet_dataloader
from collections import OrderedDict
import pickle
import datetime
try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes

# import deepspeed

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

# from mmdet\models\detectors\base.py
def base_parse_losses(losses):
    """Parse the raw outputs (losses) of the network.

    Args:
        losses (dict): Raw output of the network, which usually contain
            losses and other necessary information.

    Returns:
        tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
            which may be a weighted sum of all losses, log_vars contains \
            all the variables to be sent to the logger.
    """
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items()
               if 'loss' in _key)

    # If the loss_vars has different length, GPUs will wait infinitely
    if dist.is_available() and dist.is_initialized():
        log_var_length = torch.tensor(len(log_vars), device=loss.device)
        dist.all_reduce(log_var_length)
        message = (f'rank {dist.get_rank()}' +
                   f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                   ','.join(log_vars.keys()))
        assert log_var_length == len(log_vars) * dist.get_world_size(), \
            'loss log variables are different across GPUs!\n' + message

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars



def init_valid_data(cfg, distributed):
    # val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
    # if val_samples_per_gpu > 1:
    #     # Replace 'ImageToTensor' to 'DefaultFormatBundle'
    #     cfg.data.val.pipeline = replace_ImageToTensor(
    #         cfg.data.val.pipeline)
    #
    # val_dataset = build_dataset(cfg.data.val)
    # # val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    # val_dataloader = build_mmdet_dataloader(
    #     val_dataset,
    #     samples_per_gpu=val_samples_per_gpu,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=distributed,
    #     shuffle=False,
    #     runner_type=runner_type,
    #     persistent_workers=cfg.data.get('persistent_workers', False)
    # )

    dataset = build_dataset(cfg.data.val)

    test_dataloader_default_args = dict(
        samples_per_gpu=cfg.data.samples_per_gpu, workers_per_gpu=cfg.data.workers_per_gpu, dist=distributed, shuffle=False)

    # test_loader_cfg = {
    #     **test_dataloader_default_args,
    #     # {}
    #     **cfg.data.get('test_dataloader', {})
    # }

    data_loader = build_dataloader(dataset, **test_dataloader_default_args)

    return data_loader



@torch.no_grad()
def valid_data(cfg, model, data_loader, logger, epoch=0):
    model.eval()
    result_list = []

    all_count = len(data_loader)

    last_time = time.time()

    for i, batch_data_container in enumerate(data_loader):
        batch_data = parse_batch_data_container(batch_data_container)
        result = model(return_loss=False, rescale=True, **batch_data)
        # print(f'valid {i} / {len(data_loader)}')

        iter_time = time.time() - last_time
        eta_time = iter_time * (all_count - i)

        logger.info(
            f'Valid Epoch {epoch}, idx {i} / {all_count}, percent {i/ all_count:.2f}, bs {cfg.data.samples_per_gpu} , eta {datetime.timedelta(seconds=int(eta_time))} , iter_time {datetime.timedelta(seconds=int(iter_time))}')
        last_time = time.time()

        result_list.extend(result)

    print(len(result_list))

    # 先保留raw_result
    raw_valid_res_file = os.path.join(cfg.work_dir, 'raw_valid_result.pkl')
    with open(raw_valid_res_file, 'wb') as file_out:
        pickle.dump(result_list, file_out)

    # with open(raw_valid_res_file, 'rb') as pf:
    #     result_list = pickle.load(pf)

    ev_res = data_loader.dataset.evaluate(result_list, logger=logger, dump=True, dump_dir=cfg.work_dir)
    logger.info(f'Valid Epoch: {epoch}, result: {ev_res}')
    # mmcv.dump(results, args.out)
    # data_loader.format_results(outputs, **kwargs)


def parse_batch_data_container(batch_data_container):
    """
    解包 parse_batch_data_container
    Args:
        batch_data_container:

    Returns:

    """
    new_batch_data = {}
    for key, value in batch_data_container.items():

        if key == 'img_metas':
            new_batch_data[key] = value.data[0]
        elif key == 'img':
            new_batch_data[key] = value.data[0].float()
        elif 'gt' in key:
            new_batch_data[key] = [item for item in value.data[0]]

        else:
            assert 0

    return new_batch_data

def main():
    args = parse_args()

    # torch.distributed.barrier(device_ids=int(os.environ["LOCAL_RANK"]))
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')
    # datasets = [build_dataset(cfg.data.train)]

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    train_data = build_dataset(cfg.data.train)
    logger.info(f'len(train_data) : {len(train_data)}')
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_data_loader = build_mmdet_dataloader(
        train_data,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=cfg.data.get('persistent_workers', False))

    val_dataloader = init_valid_data(cfg, distributed)

    gradient_accumulation_steps = 1

    ds_config = {
        "train_micro_batch_size_per_gpu": cfg.data.samples_per_gpu,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {
                "device": "cpu"
            }
        }
    }



    for epoch in range(0, cfg.runner.max_epochs):
        all_count = len(train_data_loader)
        iter_count = all_count // gradient_accumulation_steps
        iter = 0
        real_batch_size = gradient_accumulation_steps * cfg.data.samples_per_gpu
        last_time = time.time()
        for idx, batch_data in enumerate(train_data_loader):

            new_batch_data = parse_batch_data_container(batch_data)

            losses = model(**new_batch_data)

            loss, log_vars = base_parse_losses(losses)
            # outputs = dict(
            #     loss=loss, log_vars=log_vars, num_samples=len(new_batch_data['img_metas']))
            #
            # optimizer.zero_grad()
            # outputs['loss'].backward()
            # optimizer.step()

            loss /= gradient_accumulation_steps
            loss.backward()
            if ((idx + 1) % gradient_accumulation_steps == 0) or (idx + 1 == all_count):
                optimizer.step()
                optimizer.zero_grad()

                iter_time = time.time() - last_time
                eta_time = iter_time * (iter_count - iter)
                logger.info(f"Epoch {epoch}, idx {idx} / {all_count}, iter {iter} / {iter_count}, bs {cfg.data.samples_per_gpu} *acc {gradient_accumulation_steps}: {real_batch_size}, eta {datetime.timedelta(seconds=int(eta_time))}, iter_time {datetime.timedelta(seconds=int(iter_time))}, loss {loss.item():.4f}, log_vars : {log_vars}")
                iter += 1
                last_time = time.time()

        check_pt_file = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, check_pt_file)

        #  every epoch valid data
        valid_data(cfg, model, val_dataloader, logger, epoch)


    # if len(cfg.workflow) == 2:
    #     val_dataset = copy.deepcopy(cfg.data.val)
    #     # in case we use a dataset wrapper
    #     if 'dataset' in cfg.data.train:
    #         val_dataset.pipeline = cfg.data.train.dataset.pipeline
    #     else:
    #         val_dataset.pipeline = cfg.data.train.pipeline
    #     # set test_mode=False here in deep copied config
    #     # which do not affect AP/AR calculation later
    #     # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
    #     val_dataset.test_mode = False
    #     datasets.append(build_dataset(val_dataset))
    # if cfg.checkpoint_config is not None:
    #     # save mmdet version, config file content and class names in
    #     # checkpoints as meta data
    #     cfg.checkpoint_config.meta = dict(
    #         mmdet_version=mmdet_version,
    #         mmseg_version=mmseg_version,
    #         mmdet3d_version=mmdet3d_version,
    #         config=cfg.pretty_text,
    #         CLASSES=datasets[0].CLASSES,
    #         PALETTE=datasets[0].PALETTE  # for segmentors
    #         if hasattr(datasets[0], 'PALETTE') else None)
    # # add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES
    # train_model(
    #     model,
    #     datasets,
    #     cfg,
    #     distributed=distributed,
    #     validate=(not args.no_validate),
    #     timestamp=timestamp,
    #     meta=meta)


if __name__ == '__main__':
    main()
