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
    _,test_loader = get_dataloader(args)

    '''valuation'''
    if args.mod == 'sam_adpt':
        net.eval()
        tol, (edice,eaji,epq) = function.validation_sam(args, test_loader, start_epoch, net)
        logger.info(f'Total score: {tol}, DICE: {edice},AJI: {eaji},PQ:{epq} || @ epoch {start_epoch}.')
    #DSB     DICE: 0.93091324829536935,AJI: 0.756618987885321375,PQ:0.74384320482745632
    #MoNuSeg DICE: 0.85703477632958764,AJI: 0.508903847576684113,PQ:0.50659374751029468
    #GlaS    DICE: 0.91927013723857629,AJI: 0.736702857672964612,PQ:0.72918243766949212
if __name__ == '__main__':
    main()
