import os
import random
import yaml
import argparse
import pdb
import numpy as np
import torch
from easydict import EasyDict
from model.trainer_mae import Trainer_mae
from model.trainer_fm import Trainer_fm

def parse_args():
    parser = argparse.ArgumentParser(description='implementation of network')
    parser.add_argument('--config_dir', default='')

    parser.add_argument('--data_root', type=str, default='', help='dataset dir')
    parser.add_argument('--save_path', type=str, default='', help='save dir')
    parser.add_argument('--log_dir', type=str, default='', help='log dir')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='pretrained weights')

    return parser.parse_args()

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def main():
    args = parse_args()
    with open(args.config_dir) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v

    config = EasyDict(config)

    seed = config.seed
    set_all_seeds(seed) 

    #model = Trainer_mae(config)
    model = Trainer_fm(config)

    if config.eval_mode:
        pass
    elif config.train:
        model.train()
    elif config.test:
        model.test()

if __name__ == "__main__":
    main()