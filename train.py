import torch
import torch.nn.parallel.data_parrllel as parallel
import torch.nn as nn
import argparse
import os
import torch.optim as optim
import time

from torch.utils.data import DataLoadaer
from hparams import hparams as hp
from dataset import lpcnetDataset
from models.model import Model

def create_model(hp):
    model = Model(
        nb_mels=hp.nb_mels,
        rnn_units1=hp.rnn_units1,
        rnn_units2=hp.rnn_units2,
        embed_size=hp.embed_size,
        frame_size=hp.frame_size,
        num_mixture=hp.num_mixture, 
        training=True)
    return model

def train(args):
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_dataset = lpcnetDataset(args)
    device = torch.device('cuda:0' if args.use_cuda else 'cpu')
    model = create_model(hp)
 
    print(model)
    
    num_gpu = torch.cuda.device_count() if args.use_cuda else 1
    
    model.train(mode=True)
    
    global_step = 0
    
    parameters = list(model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    writer = SummaryWriter(args.checkpoint_dir)

    model.to(device)

    if args.resume is not None:
        restore_step = attempt_to_restore(model, optimizer, args.resume, args.use_cuda)
        if args.keep_training:
            global_step = restore_step
        else:
            global_step = 0
     

