"""Training entry point for PID2Graph, adapted from the scene_graph branch's train.py.

Differences from the original (VG-oriented) train.py:
  - Loads PID2GraphDataset (pid2graph_data/dataset_pid2graph.py) instead of
    build_scene_graph_data - no Visual Genome HDF5/proposal/frequency-bias files
    required, PID2Graph's own GraphML + image pairs are read directly.
  - Fixes a real bug in the original: `freq_baseline.named_parameters()` at the
    optimizer's param_dicts construction is unconditional even though `freq_baseline`
    is only ever assigned `if config.MODEL.DECODER.FREQ_BIAS` - this NameErrors the
    moment FREQ_BIAS is False, which it always is here (no VG frequency-prior file
    exists for PID2Graph). Guarded the same way the rest of the file already guards
    every other freq_baseline use.
  - Everything else (model build, matcher, loss, trainer/evaluator wiring, checkpoint
    resume) is unchanged from the original, since none of that is VG-specific.
"""
from argparse import ArgumentParser
import os
import yaml
import json
from shutil import copyfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from monai.data import DataLoader, DistributedSampler

from pid2graph_data.dataset_pid2graph import build_pid2graph_data
from utils import image_graph_collate_scene_graph
from trainer import build_trainer
from models import build_model
from models.matcher_scene import build_matcher
from losses import SetCriterion

import torch.distributed as dist
import ignite.distributed as igdist
from ignite.contrib.handlers.tqdm_logger import ProgressBar

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, required=True,
                         help='config file (.yml) containing the hyper-parameters for training.')
    parser.add_argument('--resume', default='', type=str, help='checkpoint of the last epoch of the model')
    parser.add_argument('--device', default='cuda', help='device to use for training')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--nproc_per_node", default=None, type=int)
    parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=None)
    parser.add_argument('--debug', dest='debug', help='do fast debug (tiny subset)', action='store_true')
    parser.add_argument('--exp_name', dest='exp_name', help='name of the experiment', type=str, required=True)
    parser.add_argument('-b', dest='batch_size', help='batch size', type=int, default=2)
    parser.add_argument('--eval', dest='eval', help='evaluate pretrained model', action='store_true')
    return parser.parse_args()


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def match_name_keywords(n, name_keywords):
    return any(b in n for b in name_keywords)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(rank, args):
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)
    config.MODEL.RESUME = args.resume
    config.log.exp_name = args.exp_name
    config.DATA.BATCH_SIZE = args.batch_size

    print('Experiment Name : ', config.log.exp_name)
    print('Batch size : ', config.DATA.BATCH_SIZE)
    if args.debug:
        config.DATA.NUM_WORKERS = 0

    exp_path = os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED))
    if os.path.exists(exp_path) and args.resume == '':
        print('WARNING: Experiment folder exists, please change exp name in config file')
    elif not args.eval and not len(config.MODEL.RESUME) > 0:
        os.makedirs(exp_path, exist_ok=True)
        copyfile(args.config, os.path.join(exp_path, "config.yaml"))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    args.distributed = False
    args.rank = rank
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        print('Running Distributed:', args.distributed, '; GPU:', args.gpu, '; RANK:', args.rank)

    if igdist.get_local_rank() > 0:
        igdist.barrier()

    train_ds, val_ds = build_pid2graph_data(config, mode='split')

    if args.debug:
        train_ds.samples = train_ds.samples[:30]
        val_ds.samples = val_ds.samples[:30]

    if igdist.get_local_rank() == 0:
        igdist.barrier()

    train_loader = igdist.auto_dataloader(
        train_ds, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
        collate_fn=image_graph_collate_scene_graph, pin_memory=True, shuffle=True,
    )
    val_loader = igdist.auto_dataloader(
        val_ds, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS,
        collate_fn=image_graph_collate_scene_graph, pin_memory=True, shuffle=False,
    )

    device = torch.device(args.device)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        args.rank = igdist.get_rank()
        device = torch.device(f"cuda:{args.rank}")

    model = build_model(config)
    print('Number of parameters : ', count_parameters(model))
    # No freq_baseline for PID2Graph - config.MODEL.DECODER.FREQ_BIAS must be False
    # (no VG-style frequency-prior file exists for this dataset).
    assert not config.MODEL.DECODER.FREQ_BIAS, "FREQ_BIAS requires a VG-style frequency file this dataset doesn't have; set it False in the config"
    freq_baseline = None

    net_wo_dist = model.to(device)
    relation_embed = model.relation_embed.to(device)

    model = igdist.auto_model(model)
    relation_embed = igdist.auto_model(relation_embed)

    if args.distributed:
        net_wo_dist = model.module

    matcher = build_matcher(config=config)
    loss = SetCriterion(
        config, matcher, relation_embed, freq_baseline=None,
        use_target=True, focal_alpha=config.TRAIN.FOCAL_LOSS_ALPHA,
    ).to(device)

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not match_name_keywords(n, ["backbone", 'reference_points', 'sampling_offsets'])
                       and p.requires_grad],
            "lr": float(config.TRAIN.LR),
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, ["backbone"]) and p.requires_grad],
            "lr": float(config.TRAIN.LR_BACKBONE),
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, ['reference_points', 'sampling_offsets']) and p.requires_grad],
            "lr": float(config.TRAIN.LR) * 0.1,
        },
        # No freq_baseline param group - unlike the original train.py, which
        # unconditionally iterates freq_baseline.named_parameters() here even when
        # freq_baseline was never assigned (a real NameError bug upstream when
        # FREQ_BIAS is False, which it always is for this dataset).
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=float(config.TRAIN.LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY))
    optimizer = igdist.auto_optim(optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAIN.LR_DROP)

    if len(config.MODEL.RESUME) > 1 or len(config.MODEL.PRETRAIN) > 1:
        assert not (len(config.MODEL.RESUME) > 0 and len(config.MODEL.PRETRAIN) > 0), \
            'Both pretrain and resume cant be used together'
        ckpt_path = config.MODEL.RESUME if len(config.MODEL.RESUME) > 0 else config.MODEL.PRETRAIN
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if len(config.MODEL.PRETRAIN) > 0:
            keys_to_alter = [x for x in list(checkpoint['model'].keys()) if 'class_embed' in x or 'bbox_embed' in x]
            for key in keys_to_alter:
                if 'class' in key:
                    del checkpoint['model'][key]
                if 'bbox' in key:
                    checkpoint['model']['transformer.decoder.' + key] = checkpoint['model'][key]
            with torch.no_grad():
                if config.MODEL.DECODER.NUM_QUERIES == 300 + 1:
                    net_wo_dist.query_embed.weight[:-1, :] = checkpoint['model']['query_embed.weight']
            del checkpoint['model']['query_embed.weight']
            missing_keys, unexpected_keys = net_wo_dist.load_state_dict(checkpoint['model'], strict=False)
        else:
            missing_keys, unexpected_keys = net_wo_dist.load_state_dict(checkpoint['net'], strict=False)
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if len(config.MODEL.RESUME) > 0 and not args.eval:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            last_epoch = scheduler.last_epoch

    if not args.eval:
        writer = SummaryWriter(log_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED)))
    else:
        writer = None

    from evaluator_sg import build_evaluator

    evaluator = build_evaluator(
        val_loader, model, optimizer, scheduler, writer, config, device,
        coco_evaluator=None, debug=args.debug, distributed=args.distributed,
        local_rank=args.rank, freq_baseline=None,
    )
    trainer = build_trainer(
        train_loader, model, loss, optimizer, scheduler, writer, evaluator, config, device,
        distributed=args.distributed, local_rank=args.rank,
    )
    if dist.get_rank() == 0:
        pbar = ProgressBar()
        if args.eval:
            pbar.attach(evaluator)
        else:
            pbar.attach(trainer, output_transform=lambda x: {'loss': x["loss"]["total"]})

    if len(config.MODEL.RESUME) > 0 and not args.eval:
        evaluator.state.epoch = last_epoch
        trainer.state.epoch = last_epoch
        trainer.state.iteration = trainer.state.epoch_length * last_epoch

    if args.eval:
        evaluator.run()
    else:
        trainer.run()
    dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    if args.cuda_visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    with igdist.Parallel(backend='gloo', nproc_per_node=args.nproc_per_node) as parallel:
        parallel.run(main, args)
