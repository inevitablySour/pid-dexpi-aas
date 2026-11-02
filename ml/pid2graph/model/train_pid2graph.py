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
    parser.add_argument('--debug', dest='debug', help='do fast debug (tiny subset, 2 epochs unless --epochs)', action='store_true')
    parser.add_argument('--epochs', dest='epochs', help='override TRAIN.EPOCHS from the config', type=int, default=None)
    parser.add_argument('--init_from', default='', type=str,
                        help='start from this checkpoint\'s weights only (fresh optimizer, scheduler and epoch '
                             'count) - for fine-tuning a finished run, unlike --resume which continues its schedule')
    parser.add_argument('--freeze_detector', action='store_true',
                        help='train only the relation (connection) head; backbone, encoder, decoder and the '
                             'symbol class/box heads stay fixed')
    parser.add_argument('--lr', dest='lr', type=float, default=None, help='override TRAIN.LR')
    parser.add_argument('--exp_name', dest='exp_name', help='name of the experiment', type=str, required=True)
    parser.add_argument('-b', dest='batch_size', help='batch size', type=int, default=2)
    parser.add_argument('--eval', dest='eval', help='evaluate pretrained model', action='store_true')
    parser.add_argument('--eval_split', choices=['val', 'test', 'team_test'], default='val',
                        help="with --eval: 'test' = held-out OPEN100 sheets (the fold's 3 with --fold), "
                             "'team_test' = the held-out team drawing (DATA.TEAM_TEST_SUBSETS)")
    parser.add_argument('--fold', type=int, default=None,
                        help='OPEN100 cross-validation fold (0-3): 8 real sheets join training, 1 validation, '
                             '3 are this fold\'s test set; also adds DATA.EXTRA_TRAIN_SUBSETS to training. '
                             '-1 = the final model: all 12 real sheets and every team drawing (incl. '
                             'DATA.TEAM_TEST_SUBSETS) train, nothing is held out')
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
    if args.lr is not None:
        config.TRAIN.LR = args.lr
    assert not (args.init_from and args.resume), '--init_from and --resume are mutually exclusive'

    print('Experiment Name : ', config.log.exp_name)
    print('Batch size : ', config.DATA.BATCH_SIZE)
    if args.epochs is not None:
        config.TRAIN.EPOCHS = args.epochs
    elif args.debug:
        # The config's 150 epochs would make a "quick" debug/benchmark run take hours.
        config.TRAIN.EPOCHS = 2
    if args.debug:
        config.DATA.NUM_WORKERS = 0

    exp_path = os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED))
    if os.path.exists(exp_path) and args.resume == '':
        print('WARNING: Experiment folder exists, please change exp name in config file')
    elif not args.eval and not len(config.MODEL.RESUME) > 0:
        os.makedirs(exp_path, exist_ok=True)
        copyfile(args.config, os.path.join(exp_path, "config.yaml"))

    # benchmark=True autotunes per input shape; random resize/crop makes nearly every
    # shape new, so it never stops tuning - measured 1.4-1.8 s vs 0.10 s per sample.
    torch.backends.cudnn.benchmark = False
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

    train_ds, val_ds = build_pid2graph_data(config, mode='split', fold=args.fold)
    if args.eval and args.eval_split in ('test', 'team_test'):
        val_ds = build_pid2graph_data(config, mode=args.eval_split, fold=args.fold)
    print(f'Train patches: {len(train_ds)}  val patches: {len(val_ds)}' + (f'  (fold {args.fold})' if args.fold is not None else ''))

    if args.debug:
        train_ds.samples = train_ds.samples[:30]
        val_ds.samples = val_ds.samples[:30]

    if igdist.get_local_rank() == 0:
        igdist.barrier()

    if getattr(config.DATA, 'BALANCED_SAMPLING', False) and not args.eval:
        # Rare classes and the scarce real/team patches are drawn more often; an "epoch"
        # is DATA.EPOCH_SAMPLES draws (with replacement) rather than one pass over 30k
        # mostly-synthetic patches. See sample_weights() for the weighting.
        from pathlib import Path as _Path
        from torch.utils.data import WeightedRandomSampler
        from pid2graph_data.dataset_pid2graph import sample_weights
        weights = sample_weights(train_ds.samples, config, _Path(config.DATA.DATA_PATH) / '.class_counts_cache.json')
        n_draw = len(train_ds) if args.debug else int(getattr(config.DATA, 'EPOCH_SAMPLES', len(train_ds)))
        sampler = WeightedRandomSampler(weights, num_samples=n_draw, replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS,
            collate_fn=image_graph_collate_scene_graph, pin_memory=True,
        )
        print(f'Balanced sampling: {n_draw} draws per epoch')
    else:
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

    if args.init_from:
        state = torch.load(args.init_from, map_location='cpu')['net']
        own = net_wo_dist.state_dict()
        # A layer whose shape changed (e.g. the relation head's output when NUM_REL_CLS
        # goes 1 -> 2 for process vs signal lines) keeps its fresh initialisation.
        # The class head growing (7 symbol classes -> + bend, junction for DATA.LINE_POINTS)
        # keeps its learned rows - background and the 7 symbols come first - and only the
        # new classes start fresh.
        n_old = state['class_embed.0.weight'].shape[0] if 'class_embed.0.weight' in state else None
        n_new = own['class_embed.0.weight'].shape[0] if 'class_embed.0.weight' in own else None
        if n_old and n_new and n_new > n_old:
            for k, v in list(state.items()):
                if k.startswith('class_embed.') and k in own and own[k].shape[1:] == v.shape[1:]:
                    grown = own[k].detach().cpu().clone()  # the model is on the GPU, the checkpoint on the CPU
                    grown[:v.shape[0]] = v
                    state[k] = grown
            # The relation head reads each item's class scores too (ADD_EMB_REL: per item
            # [features, class scores, box]), so its input widens: the learned weights move
            # with their inputs, the new classes' scores start with no say.
            if config.MODEL.DECODER.ADD_EMB_REL:
                hid = int(config.MODEL.DECODER.HIDDEN_DIM)
                per_old, per_new = hid + n_old + 4, hid + n_new + 4
                where = [part * per_new + o for part in (0, 1) for o in list(range(hid + n_old)) + [hid + n_new + b for b in range(4)]]
                where += [2 * per_new + o for o in range(hid)]  # the relation token
                where = torch.tensor(where)
                for k, axis in (('relation_embed.layers.0.weight', 0), ('relation_embed.layers.0.bias', 0),
                                ('relation_embed.layers.1.weight', 1)):
                    if k in state and k in own and state[k].shape[axis] == 2 * per_old + hid:
                        grown = own[k].detach().cpu().clone()  # the model is on the GPU, the checkpoint on the CPU
                        if k == 'relation_embed.layers.1.weight':
                            grown.zero_()
                            grown[:, where] = state[k]
                        else:
                            grown[where] = state[k]
                        state[k] = grown
            print('Classes grown', n_old - 1, '->', n_new - 1, '(learned weights kept)')
        reshaped = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
        state = {k: v for k, v in state.items() if k not in reshaped}
        missing_keys, unexpected_keys = net_wo_dist.load_state_dict(state, strict=False)
        assert set(missing_keys) == set(reshaped) and not unexpected_keys, (missing_keys, unexpected_keys)
        print('Initialised weights from', args.init_from, '| re-initialised (shape changed):', reshaped or 'none')
    if args.freeze_detector:
        # Only the connection head learns. Used to fix connection labels without
        # disturbing symbol detection, which was trained on correct labels.
        for name, p in net_wo_dist.named_parameters():
            p.requires_grad = name.startswith('relation_embed')
        print('Trainable parameters (relation head only):', count_parameters(net_wo_dist))

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

    param_dicts = [g for g in param_dicts if g["params"]]  # frozen groups are empty
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

    # Not evaluator_sg: that one reports VG scene-graph recall@K and needs
    # config.DATA.LABEL_DATA_DIR (a Visual Genome idx_to_predicate json). This one
    # reports COCO node mAP + edge mAP, comparable to the published P&ID baselines.
    from evaluator_pid2graph import build_evaluator

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
    # No dist.destroy_process_group() here: igdist.Parallel.__exit__ already does it, and
    # calling it twice raised AssertionError at the end of every run.


if __name__ == '__main__':
    args = parse_args()
    if args.cuda_visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    with igdist.Parallel(backend='gloo', nproc_per_node=args.nproc_per_node) as parallel:
        parallel.run(main, args)
