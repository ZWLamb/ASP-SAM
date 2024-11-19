
import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import cfg_dsb as cfg  ####
#from models.discriminatorlayer import discriminator
from conf import settings
from utils import vis_image,eval_seg

# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def mse_loss(pred, true):

    loss = pred - true
    loss = (loss * loss).mean()
    return loss

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, scheduler=None, vis = 50):
    ind = 0
    # Training mode
    net.train()
    optimizer.zero_grad()
    lossfunc = criterion_G

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))


    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            # torch.cuda.empty_cache()
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)#8*3*256*256
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)#8*1*256*256
            sc_mask = pack['scale_mask'].to(dtype = torch.float32, device = GPUdevice)#8*256*256
            pt = pack['pt']
            showp = pt
            point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            #prompts = torchvision.transforms.Resize((64, 64))(sc_mask)#8*1*256*256
            ind += 1

            if point_labels.clone().flatten()[0] != -1:
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                if(len(point_labels.shape)==1): # only one point prompt
                    coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                pt = (coords_torch, labels_torch)


            '''Train'''
            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters():
                    if "Adapter" not in n:
                        value.requires_grad = False
                    else:
                        value.requires_grad = True

            else:
                for n, value in net.image_encoder.named_parameters(): 
                    value.requires_grad = True
                    
            imge= net.image_encoder(imgs)

            prompts,_ = net.prompt_gen(imgs) #TODO:对_做监督 或者 修改L136 masks的维度输入

            for n, value in net.prompt_gen.named_parameters():
                if 'backbone' in n:
                    value.requires_grad = True
                else:
                    value.requires_grad = False

            with torch.no_grad():
                if args.net == 'sam' or args.net == 'mobile_sam':
                    se, de = net.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None,
                        scale_masks= prompts,#8*1*64*64
                    )

            if args.net == 'sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=(args.multimask_output > 1),
                )
            elif args.net == 'mobile_sam':
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=(args.multimask_output > 1),
                )
            elif args.net == "efficient_sam":
                se = se.view(
                    se.shape[0],
                    1,
                    se.shape[1],
                    se.shape[2],
                )
                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    multimask_output=False,
                )
                
            # Resize to the ordered output size
            pred = F.interpolate(pred,size=(args.out_size,args.out_size))
            pred = F.interpolate(pred[:,:2,:,:],size=(args.out_size,args.out_size))

            prompts = F.interpolate(prompts,size=(args.out_size,args.out_size))

            loss_prompt = mse_loss(prompts,sc_mask) #prompt loss
            loss_mask = lossfunc(pred, masks)

            loss = 1.0*loss_mask + 1.0*loss_prompt

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)

            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()



            # 输出当前学习率
            # current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率
            # print(f'Epoch {epoch + 1}/{epoch}, Learning Rate: {current_lr:.6f}')

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name[:2]:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                    vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()

        # scheduler.step(loss_mask)

    return loss

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    # eval mode
    net.eval()
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*3
    tot = 0
    threshold = (1)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            name = pack['name']
            ipt_imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)    #0~255
            nuclei_masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
            inst_mask = pack['inst_mask'].to(dtype = torch.float32, device = GPUdevice)
            #sc_mask = pack['scale_mask'].to(dtype = torch.float32, device = GPUdevice) #8*256*256
            ptw = pack['pt']
            point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']
            buoy = 0

            evl_ch = int(ipt_imgs.size(-1)) #256
            while (buoy + evl_ch) <= ipt_imgs.size(-1):

                pt = ptw
                imgs = ipt_imgs[...,buoy:buoy + evl_ch]
                masks = nuclei_masks[...,buoy:buoy + evl_ch]
                buoy += evl_ch
                showp = pt
                mask_type = torch.float32
                ind += 1

                if point_labels.clone().flatten()[0] != -1:
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if(len(point_labels.shape)==1): # only one point prompt
                        coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                    pt = (coords_torch, labels_torch) #pt prompt 提示信息 点击坐标,标签

                imgs = imgs.to(dtype = mask_type,device = GPUdevice)
                #prompts = torchvision.transforms.Resize((64, 64))(sc_mask)  # 8*1*256*256

                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(imgs)#3*252*256 -> 256*16*16
                    prompts, _ = net.prompt_gen(imgs)
                    if args.net == 'sam' or args.net == 'mobile_sam':
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                            scale_masks = prompts
                        )# 1*3*256   1*256*64*64

                    if args.net == 'sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=(args.multimask_output > 1),
                        )
                    elif args.net == 'mobile_sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,#1*3*256 #稀疏
                            dense_prompt_embeddings=de, #1*256*64*64 #稠密
                            multimask_output=False,
                        )
                    elif args.net == "efficient_sam":
                        se = se.view(
                            se.shape[0],
                            1,
                            se.shape[1],
                            se.shape[2],
                        )
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=se,
                            multimask_output=False,
                        )

                    # Resize to the ordered output size
                    #pred = F.interpolate(pred[:,:2,:,:],size=(args.out_size,args.out_size))
                    tot += lossfunc(pred, masks)

                    '''vis images'''
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name[:2
                        
                        ]:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'
                        vis_image(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)
                    

                    temp = eval_seg(pred, masks, threshold,inst_mask,name)  #计算指标 返回三个指标
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    # if args.evl_chunk:
    #     n_val = n_val * (ipt_imgs.size(-1) // evl_ch)

    return tot/ n_val , tuple([a/n_val for a in mix_res])



