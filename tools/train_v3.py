from __future__ import absolute_import
import argparse
import json
import os
import os.path as osp
import numpy as np
import sys
import time
import random
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='', help='path to config file')
    parser.add_argument('--dropout', type=float, default=0.2, help='path to config file')
    parser.add_argument('--gpu-devices', '-g', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    os.environ['OMP_NUM_THREADS'] = '8'

    sys.path.append('.')
    from data import init_dataset
    from losses import BPRLoss, Triplet
    from models import build_model
    from engine import Trainer_v3
    from solver import build_optimizer
    from utils import set_random_seed, Logger, collect_env_info
    from config import cfg, model_kwargs, data_kwargs

    cfg.TRAIN.DROPOUT = args.dropout

    if args.config:
        cfg.merge_from_file(args.config)
    
    set_random_seed(cfg.TRAIN.SEED)

    cfg.LOG_DIR = osp.join(cfg.LOG_ROOT, cfg.DIR_NAME, time.strftime('%Y%m%d-%H%M%S'))
    cfg.CHECKPOINTS_DIR = osp.join(cfg.CHECKPOINTS_ROOT, cfg.DIR_NAME, time.strftime('%Y%m%d-%H%M%S'))
    log_name = 'train.txt'
    sys.stdout = Logger(osp.join(cfg.LOG_DIR, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    dataset = init_dataset(cfg.DATA.DATASET, **data_kwargs(cfg))

    print('Building model: {}'.format(cfg.MODEL.NAME))
    model = build_model(cfg.MODEL.NAME, **model_kwargs(cfg))
    print('Model complexity: params={:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print(model)

    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    optimizer = build_optimizer(cfg, model)
    loss = Triplet()

    trainer = Trainer_v3(cfg, model, optimizer, loss, dataset)
    trainer.run()


if __name__ == '__main__':
    main()
