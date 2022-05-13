# coding=utf-8
# Copy from lxmert with modifications

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    # A fake args object
    # .. so that I can use argparser for the upstream pipeline
    class vqa_args:
        def __init__(self):
            self.model = 'lxmert'
            self.train = 'train,nominival'
            self.valid = 'minival'
            self.test = None
            self.batch_size = 32
            self.optim = 'bert'
            self.lr = 1e-4
            self.epochs = 2
            self.dropout = 0.1
            self.seed = 9595
            self.max_seq_length = 20
            self.output = 'models/trained/'
            self.fast = False
            self.tiny = False
            self.tqdm = True
            self.load_trained = None
            self.load_pretrained = None 
            self.from_scratch = None
            self.mce_loss = False
            self.multiGPU = False
            self.num_workers = 0
            self.optimizer = get_optimizer('bert')
    
    args = vqa_args()

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
