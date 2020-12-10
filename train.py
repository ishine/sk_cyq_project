import torch
import torch.nn.parallel.data_parrllel as parallel
import torch.nn as nn
import argparse
import os
import torch.optim as optim
import time

from torch.utils.data import DataLoadaer

from utils.dataset import lpcnetDataset
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

def save_checkpoint(args, model, optimizer, step):
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.ckpt-{}.pt".format(step))

    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": step
                }, checkpoint_path)

    print("Saved checkpoint: {}".format(checkpoint_path))

    with open(os.path.join(args.checkpoint_dir, 'checkpoint'), 'w') as f:
        f.write("model.ckpt-{}.pt".format(step))

def attempt_to_restore(model, optimizer, checkpoint_dir, use_cuda):
    checkpoint_list = os.path.join(checkpoint_dir, 'checkpoint')

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(
            checkpoint_dir, "{}".format(checkpoint_filename))
        print("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]

    else:
        global_step = 0

    return global_step

def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)

    return checkpoint

def train(args):
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    hparams = YParams(args.yaml_conf)
    hp = hparams.parse(args.hparams) 

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
    t_start = hp.t_start
    t_end   = hp.t_end
    interval = hp.interval
    density = hp.density

    for epoch in range(args.epochs):
        train_data_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)

        for batch, (in_data, mel, output) in enumerate(train_data_loader):
            start = time.time()
            if num_gpu > 1:
                output_predict = parallel(model, (in_data, mel))
            else:
                output_predict = model(in_data, mel) 
            
            loss = discretized_mix_logistic_loss(output, output_predict, log_scales).to(device)   
         
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', type=str, default='', help='Directory of file list')
    parser.add_argument('--mel_dir', type=str, default='', help='Directory of mel')
    parser.add_argument('--pcm_dir', type=str, default='', help='Directory of pcm')
    parser.add_argument('--num_workers',type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--checkpoint_dir', type=str, default="logdir", help="Directory to save model")
    parser.add_argument('--resume', type=str, default=None, help="The model name to restore")
    parser.add_argument('--checkpoint_step', type=int, default=5000)
    parser.add_argument('--summary_step', type=int, default=100)
    parser.add_argument('--use_cuda', type=_str_to_bool, default=True)
    parser.add_argument('--keep_training', type=_str_to_bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--warmup_steps', type=int, default=80000)
    parser.add_argument('--decay_learning_rate', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--yaml_conf', default='hparams.yaml', help='yaml files for configurations.')
    parser.add_argument('--hparams', default='')

    args = parser.parse_args()
    train(args)
