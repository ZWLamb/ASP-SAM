# train_dsb.py

import os
import time
from datetime import datetime
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import cfg_dsb
import function
from conf import settings
from dataset import *
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():

    args = cfg_dsb.parse_args()

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    if args.pretrain:#False
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights,strict=False)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.98, verbose=True, patience=5, eps=1e-8, threshold=1e-20)


    '''load pretrained model'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']

        net.load_state_dict(checkpoint['state_dict'],strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    train_loader, test_loader = get_dataloader(args)

    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    '''begin Training'''
    best_dice = 0.0
    for epoch in range(settings.EPOCH):
        if epoch and epoch < 5:
           tol, (eiou, edice, eaji) = function.validation_sam(args, test_loader, epoch, net, writer)
           logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, train_loader, epoch, writer, vis = args.vis,scheduler = scheduler)

        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:

            tol, (eiou, edice,eaji) = function.validation_sam(args, test_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')


            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if edice > best_dice:
                best_dice = edice
                is_best = True

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="best_dice_checkpoint.pth")
            else:
                is_best = False

    writer.close()


if __name__ == '__main__':
    main()
