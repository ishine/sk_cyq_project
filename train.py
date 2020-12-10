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
     
    customer_optimizer = Optimizer(optimizer, args.learning_rate, global_step, args.warmup_steps, args.decay_learning_rate)
    
    log_scales = -7.0
    t_start = 2000
    t_end   = 40000
    interval = 400
    density = (0.05, 0.05, 0.2)

    for epoch in range(args.epochs):
        train_data_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)

        for batch, (in_data, mel, output) in enumerate(train_data_loader):
            start = time.time()
            if num_gpu > 1:
                output_predict = parallel(model, (in_data, mel))
            else:
                output_predict = model(in_data, mel) 
            
            loss = discretized_mix_logistic_loss(output, output_predict, log_scales)   
         
            global_step += 1
            customer_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=0.5)
            customer_optimizer.step_and_update_lr()
            model.sparse_gru_a(global_step, t_start, t_end, interval, density)
             
            print("Step: {} --loss: {:.3f} --log_scales: {:.3f} --Lr: {:g} --Time: {:.2f} seconds".format(
                   global_step, loss, log_scales, customer_optimizer.lr, float(time.time() - start)))
 
            log_scales = (-7.0 - 1.0 * global_step /50000)
            if global_step % args.checkpoint_step == 0:
                save_checkpoint(args, model, optimizer, global_step)
          
            if global_step % args.summary_step == 0:
                writer.add_scalar("loss", loss.item(), global_step)
             
