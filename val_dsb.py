#!/usr/bin/env	python3

""" valuate network using pytorch
    Lang Yi
"""

import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataset import *
from conf import settings
import time
import cfg_dsb
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function




def main():
    args = cfg_dsb.parse_args()
    GPUdevice = torch.device('cuda', args.gpu_device)
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

    '''load pretrained model'''
    assert args.weights != 0
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']

    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    '''data prepare'''
    # test_loader,_ = get_dataloader(args)
    _,test_loader = get_dataloader(args)

    '''valuation'''
    if args.mod == 'sam_adpt':
        net.eval()
        tol, (eiou, edice,eaji) = function.validation_sam(args, test_loader, start_epoch, net)
        logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice},AJI: {eaji} || @ epoch {start_epoch}.')
    #IOU: 0.8802513092095298, DICE: 0.9308728058087198,AJI: 0.7566345467884432 || @ epoch 131.
    #IOU: 0.772394071274934, DICE: 0.8623078536631456,AJI: 0.6124390383169842 || @ epoch 131.
if __name__ == '__main__':
    main()
