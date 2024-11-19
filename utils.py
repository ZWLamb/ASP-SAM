import collections
import logging
import math
import os
import pathlib
import random
import sys
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from typing import BinaryIO, List, Optional, Text, Tuple, Union

import dateutil.tz
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.utils as vutils
from monai.data import (CacheDataset, ThreadDataLoader,
                        load_decathlon_datalist, set_track_meta)
from monai.transforms import (Compose, CropForegroundd,
                              EnsureTyped, LoadImaged, Orientationd,
                              RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
                              RandShiftIntensityd, ScaleIntensityRanged,
                              Spacingd)
from PIL import Image
from torch import autograd
from torch.autograd import Function, Variable

from tqdm import tqdm
from scipy import ndimage

import cfg_dsb as cfg

args = cfg.parse_args()
device = torch.device('cuda', args.gpu_device)

import cv2

def get_network(args, net, use_gpu=True, gpu_device = 0, distribution = True):
    """
    return given network
    """

    if net == 'sam':
        from models.sam import SamPredictor, sam_model_registry
        options = ['default','vit_b','vit_l','vit_h']
        if args.encoder not in options:
            raise ValueError("Invalid encoder option. Please choose from: {}".format(options))
        else:
            net = sam_model_registry[args.encoder](args,checkpoint=args.sam_ckpt).to(device)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        if distribution != 'none':
            net = torch.nn.DataParallel(net,device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)
    return net


def get_decath_loader(args):

    train_transforms = Compose(
        [   
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_size, args.roi_size, args.chunk),
                pos=1,
                neg=1,
                num_samples=args.num_sample,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )



    data_dir = args.data_path
    split_JSON = "dataset_0.json"

    datasets = os.path.join(data_dir, split_JSON)
    datalist = load_decathlon_datalist(datasets, True, "Training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=args.b, shuffle=True)
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=2, cache_rate=1.0, num_workers=0
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    set_track_meta(False)

    return train_loader, val_loader, train_transforms, val_transforms, datalist, val_files


def cka_loss(gram_featureA, gram_featureB):

    scaled_hsic = torch.dot(torch.flatten(gram_featureA),torch.flatten(gram_featureB))
    normalization_x = gram_featureA.norm()
    normalization_y = gram_featureB.norm()
    return scaled_hsic / (normalization_x * normalization_y)

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)



@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs
) -> torch.Tensor:
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))


class RunningStats:
    def __init__(self, WIN_SIZE):
        self.mean = 0
        self.run_var = 0
        self.WIN_SIZE = WIN_SIZE

        self.window = collections.deque(maxlen=WIN_SIZE)

    def clear(self):
        self.window.clear()
        self.mean = 0
        self.run_var = 0

    def is_full(self):
        return len(self.window) == self.WIN_SIZE

    def push(self, x):

        if len(self.window) == self.WIN_SIZE:
            # Adjusting variance
            x_removed = self.window.popleft()
            self.window.append(x)
            old_m = self.mean
            self.mean += (x - x_removed) / self.WIN_SIZE
            self.run_var += (x + x_removed - old_m - self.mean) * (x - x_removed)
        else:
            # Calculating first variance
            self.window.append(x)
            delta = x - self.mean
            self.mean += delta / len(self.window)
            self.run_var += delta * (x - self.mean)

    def get_mean(self):
        return self.mean if len(self.window) else 0.0

    def get_var(self):
        return self.run_var / len(self.window) if len(self.window) > 1 else 0.0

    def get_std(self):
        return math.sqrt(self.get_var())

    def get_all(self):
        return list(self.window)

    def __str__(self):
        return "Current window values: {}".format(list(self.window))

def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2.
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times

    if pairwise_iou.size == 0:
        return 1
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred])
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union

    return aji_score
'''parameter'''


def to_valid_out(maps_f,img,seg): #multi-rater
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        maps = torch.nn.Softmax(dim = 1)(maps)
        final_seg = torch.multiply(seg,maps).sum(dim = 1, keepdim = True)
        return torch.cat((img,final_seg),1)
        # return torch.cat((img,maps),1)
    return inner

def gene_out(maps_f,img): #pure seg
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        # maps = torch.nn.Sigmoid()(maps)
        return torch.cat((img,maps),1)
        # return torch.cat((img,maps),1)
    return inner

def raw_out(maps_f,img): #raw
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        # maps = torch.nn.Sigmoid()(maps)
        return maps
        # return torch.cat((img,maps),1)
    return inner    


class CompositeActivation(torch.nn.Module):

    def forward(self, x):
        x = torch.atan(x)
        return torch.cat([x/0.67, (x*x)/0.6], 1)
        # return x


def cppn(args, size, img = None, seg = None, batch=None, num_output_channels=1, num_hidden_channels=128, num_layers=8,
         activation_fn=CompositeActivation, normalize=False, device = "cuda:0"):

    r = 3 ** 0.5

    coord_range = torch.linspace(-r, r, size)
    x = coord_range.view(-1, 1).repeat(1, coord_range.size(0))
    y = coord_range.view(1, -1).repeat(coord_range.size(0), 1)

    input_tensor = torch.stack([x, y], dim=0).unsqueeze(0).repeat(batch,1,1,1).to(device)

    layers = []
    kernel_size = 1
    for i in range(num_layers):
        out_c = num_hidden_channels
        in_c = out_c * 2 # * 2 for composite activation
        if i == 0:
            in_c = 2
        if i == num_layers - 1:
            out_c = num_output_channels
        layers.append(('conv{}'.format(i), torch.nn.Conv2d(in_c, out_c, kernel_size)))
        if normalize:
            layers.append(('norm{}'.format(i), torch.nn.InstanceNorm2d(out_c)))
        if i < num_layers - 1:
            layers.append(('actv{}'.format(i), activation_fn()))
        else:
            layers.append(('output', torch.nn.Sigmoid()))

    # Initialize model
    net = torch.nn.Sequential(OrderedDict(layers)).to(device)
    # Initialize weights
    def weights_init(module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, 0, np.sqrt(1/module.in_channels))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    net.apply(weights_init)
    # Set last conv2d layer's weights to 0
    torch.nn.init.zeros_(dict(net.named_children())['conv{}'.format(num_layers - 1)].weight)
    outimg = raw_out(lambda: net(input_tensor),img) if args.netype == 'raw' else to_valid_out(lambda: net(input_tensor),img,seg)
    return net.parameters(), outimg

def get_siren(args):
    wrapper = get_network(args, 'siren', use_gpu=args.gpu, gpu_device=torch.device('cuda', args.gpu_device), distribution = args.distributed)
    '''load init weights'''
    checkpoint = torch.load('./logs/siren_train_init_2022_08_19_21_00_16/Model/checkpoint_best.pth')
    wrapper.load_state_dict(checkpoint['state_dict'],strict=False)
    '''end'''

    '''load prompt'''
    checkpoint = torch.load('./logs/vae_standard_refuge1_2022_08_21_17_56_49/Model/checkpoint500')
    vae = get_network(args, 'vae', use_gpu=args.gpu, gpu_device=torch.device('cuda', args.gpu_device), distribution = args.distributed)
    vae.load_state_dict(checkpoint['state_dict'],strict=False)
    '''end'''

    return wrapper, vae


def siren(args, wrapper, vae, img = None, seg = None, batch=None, num_output_channels=1, num_hidden_channels=128, num_layers=8,
         activation_fn=CompositeActivation, normalize=False, device = "cuda:0"):
    vae_img = torchvision.transforms.Resize(64)(img)
    latent = vae.encoder(vae_img).view(-1).detach()
    outimg = raw_out(lambda: wrapper(latent = latent),img) if args.netype == 'raw' else to_valid_out(lambda: wrapper(latent = latent),img,seg)
    # img = torch.randn(1, 3, 256, 256)
    # loss = wrapper(img)
    # loss.backward()

    # # after much Training ...
    # # simply invoke the wrapper without passing in anything

    # pred_img = wrapper() # (1, 3, 256, 256)
    return wrapper.parameters(), outimg

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def export(tensor, img_path=None):
    # image_name = image_name or "image.jpg"
    c = tensor.size(1)
    # if c == 7:
    #     for i in range(c):
    #         w_map = tensor[:,i,:,:].unsqueeze(1)
    #         w_map = tensor_to_img_array(w_map).squeeze()
    #         w_map = (w_map * 255).astype(np.uint8)
    #         image_name = image_name[0].split('/')[-1].split('.')[0] + str(i)+ '.png'
    #         wheat = sns.heatmap(w_map,cmap='coolwarm')
    #         figure = wheat.get_figure()    
    #         figure.savefig ('./fft_maps/weightheatmap/'+str(image_name), dpi=400)
    #         figure = 0
    # else:
    if c == 3:
        vutils.save_image(tensor, fp = img_path)
    else:
        image = tensor[:,0:3,:,:]
        w_map = tensor[:,-1,:,:].unsqueeze(1)
        image = tensor_to_img_array(image)
        w_map = 1 - tensor_to_img_array(w_map).squeeze()
        # w_map[w_map==1] = 0
        assert len(image.shape) in [
            3,
            4,
        ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
        # Change dtype for PIL.Image
        image = (image * 255).astype(np.uint8)
        w_map = (w_map * 255).astype(np.uint8)

        Image.fromarray(w_map,'L').save(img_path)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None


    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output


    def close(self):
        self.hook.remove()


def hook_model(model, image_f):
    features = OrderedDict()
    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    return hook

def vis_image(imgs, pred_masks, gt_masks, save_path, reverse = False, points = None, boxes = None):
    
    b,c,h,w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    if reverse == True:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
    if c == 2: # for REFUGE multi mask output
        pred_disc, pred_cup = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), pred_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_disc, gt_cup = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), gt_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        mini_imgs = F.interpolate(imgs.clone(), size=(256, 256), mode='bilinear', align_corners=False)
        tup = (mini_imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:])
        compose = torch.cat(tup, 0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
    elif c > 2: # for multi-class segmentation > 2 classes
        preds = []
        gts = []
        for i in range(0, c):
            pred = pred_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
            preds.append(pred)
            gt = gt_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
            gts.append(gt)
        tup = [imgs[:row_num,:,:,:]] + preds + gts
        compose = torch.cat(tup,0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
    else:
        imgs = torchvision.transforms.Resize((h,w))(imgs)

        max_value = torch.max(imgs)
        if max_value > 1:
            imgs = imgs/255.0

        if imgs.size(1) == 1:
            imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        pred_masks = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_masks = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        if points != None:
            for i in range(b):
                if args.thd:
                    ps = np.round(points.cpu()/args.roi_size * args.out_size).to(dtype = torch.int)
                else:
                    ps = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
                # gt_masks[i,:,points[i,0]-5:points[i,0]+5,points[i,1]-5:points[i,1]+5] = torch.Tensor([255, 0, 0]).to(dtype = torch.float32, device = torch.device('cuda:' + str(dev)))
                for p in ps:
                    gt_masks[i,0,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.5
                    gt_masks[i,1,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.1
                    gt_masks[i,2,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.4
        if boxes is not None:
            for i in range(b):
                # the next line causes: ValueError: Tensor uint8 expected, got torch.float32
                # imgs[i, :] = torchvision.utils.draw_bounding_boxes(imgs[i, :], boxes[i])
                # until TorchVision 0.19 is released (paired with Pytorch 2.4), apply this workaround:
                img255 = (imgs[i] * 255).byte()
                img255 = torchvision.utils.draw_bounding_boxes(img255, boxes[i].reshape(-1, 4), colors="red")
                img01 = img255 / 255
                # torchvision.utils.save_image(img01, save_path + "_boxes.png")
                imgs[i, :] = img01
        tup = (imgs[:row_num,:,:,:],pred_masks[:row_num,:,:,:], gt_masks[:row_num,:,:,:])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat(tup,0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)

    return


from skimage.segmentation import watershed
def apply_watershed(distance_transform=None,#质心概率图
                    prob_map=None,#质心概率图的关键区域
                    foreground_mask=None): #语义分割连通图
    marker = ndimage.label(prob_map)[0] #质心概率图
    pred_inst_map = watershed(distance_transform,#质心概率图
                              markers=marker,      #质心概率图，阈值关键区域 index是实例索引
                              mask=foreground_mask,#语义分割连通图  用于指定分割区域的边界
                              compactness=0.01)
    return pred_inst_map


def eval_seg(pred,true_mask_p,threshold,inst_mask = None,name = None):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()

    inst_mask = inst_mask.cpu().numpy().astype('int32')

    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    elif c > 2: # for multi-class segmentation > 2 classes
        ious = [0] * c
        dices = [0] * c
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            for i in range(0, c):
                pred = vpred_cpu[:,i,:,:].numpy().astype('int32')
                mask = gt_vmask_p[:,i,:,:].squeeze(1).cpu().numpy().astype('int32')
        
                '''iou for numpy'''
                ious[i] += iou(pred,mask)

                '''dice for torch'''
                dices[i] += dice_coeff(vpred[:,i,:,:], gt_vmask_p[:,i,:,:]).item()
            
        return tuple(np.array(ious + dices) / len(threshold)) # tuple has a total number of c * 2
    else:
        eiou, edice = 0,0

        #for th in threshold:#TODO：多个阈值的平均？
        th = 1 #threshold
        gt_vmask_p = (true_mask_p == 1.0).float()
        vpred = (pred > th).float()
        vpred_cpu = vpred.cpu()

        aji_pred = pred.clone().cpu()
        aji_pred[pred<th] = 0

        disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
        disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')

        '''iou for numpy'''
        eiou += iou(disc_pred,disc_mask)

        '''dice for torch'''
        edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()

        '''aji for numpy'''
        #generate inst_pred

        eaji = 0

        for i in range(b):
            #method 1:
            # inst_pred = ndimage.label(disc_pred[i])[0]# IOU: 0.7899483984475102, DICE: 0.873596756315943,AJI: 0.6140476319989097
            #_, inst_pred = cv2.connectedComponents(cv2.convertScaleAbs(disc_pred[i]))#AJI: 0.6133659403948357
            #method 1:
            _, inst_pred, _, _ = cv2.connectedComponentsWithStats(cv2.convertScaleAbs(disc_pred[i]), connectivity=4, ltype=None)#AJI: 0.6140698584943036

            #method 2:watershed
            #
            # prob_map = aji_pred[i][0]
            #
            # prob_map = F.softmax(prob_map)
            #
            # temp_prob_map = prob_map.clone().numpy()  # 质心概率图
            # temp_nuclei_map = disc_pred[i]  # 语义分割连通图
            # temp_prob_marker = np.where(temp_prob_map > 0, 1, 0)
            # inst_pred = apply_watershed(
            #     distance_transform=temp_prob_map,  # 质心概率图
            #     prob_map=temp_prob_marker,  # 质心概率图的关键区域
            #     foreground_mask=temp_nuclei_map,  # 语义分割连通图
            # )  # 实例分割图
            #



            #th = 0.8 IOU: 0.7907680758711079, DICE: 0.8731723792517363,AJI: 0.6257697415614343 || @ epoch 131.
            #th = 0.9 IOU: 0.7906837118113254, DICE: 0.8727302849292755,AJI: 0.6298587898374084 || @ epoch 131.
            #th = 2.5 IOU: 0.753102508895277, DICE: 0.8425738338186184, AJI: 0.6708913411206191 | |
            #LiteMLA relu    #ConvLayer gelu
            #th = 1.5 Total score: 0.128044992685318, IOU: 0.7055670017266767, DICE: 0.8069948084824732,AJI: 0.5777222008821032 || @ epoch 200.
            #msa+gelu
            #
            #msa+relu+lr:fail 学习率每10轮下降策略不行
            #msa+gelu+lr:fail
            #msa+gelu+shortcut+lr


            i_pred = remap_label(inst_pred, by_size=False)
            i_mask = remap_label(inst_mask[i], by_size=False)
            aji = get_fast_aji(i_mask, i_pred)
            # print(name,":",aji)
            eaji += aji
            # eaji += get_fast_aji(i_mask, i_mask)



        #return eiou / len(threshold), edice / len(threshold), eaji, #b  #/ len(threshold)
        return eiou,edice,eaji

# @objectives.wrap_objective()
def dot_compare(layer, batch=1, cossim_pow=0):
  def inner(T):
    dot = (T(layer)[batch] * T(layer)[0]).sum()
    mag = torch.sqrt(torch.sum(T(layer)[0]**2))
    cossim = dot/(1e-6 + mag)
    return -dot * cossim ** cossim_pow
  return inner

def init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



def update_d(args, netD, optimizerD, real, fake):
    criterion = nn.BCELoss()

    label = torch.full((args.b,), 1., dtype=torch.float, device=device)
    output = netD(real).view(-1)
    # Calculate loss on all-real batch
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    label.fill_(0.)
    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    return errD, D_x, D_G_z1

def calculate_gradient_penalty(netD, real_images, fake_images):
    eta = torch.FloatTensor(args.b,1,1,1).uniform_(0,1)
    eta = eta.expand(args.b, real_images.size(1), real_images.size(2), real_images.size(3)).to(device = device)

    interpolated = (eta * real_images + ((1 - eta) * fake_images)).to(device = device)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(
                                prob_interpolated.size()).to(device = device),
                            create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return grad_penalty


def random_click(mask, point_labels = 1):
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
    # max agreement position
    indices = np.argwhere(mask == max_label) 
    return point_labels, indices[np.random.randint(len(indices))]


def generate_click_prompt(img, msk, pt_label = 1):
    # return: prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:,0,:,:,:]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j,:,:,i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype = torch.float)
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0)
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1)
    msk = torch.stack(msk_list, dim=-1)

    msk = msk.unsqueeze(1)

    return img, pt, msk #[b, 2, d], [b, c, h, w, d]


def random_box(multi_rater):
    max_value = torch.max(multi_rater[:,0,:,:], dim=0)[0]
    max_value_position = torch.nonzero(max_value)

    x_coords = max_value_position[:, 0]
    y_coords = max_value_position[:, 1]


    x_min = int(torch.min(x_coords))
    x_max = int(torch.max(x_coords))
    y_min = int(torch.min(y_coords))
    y_max = int(torch.max(y_coords))


    x_min = random.choice(np.arange(x_min-10,x_min+11))
    x_max = random.choice(np.arange(x_max-10,x_max+11))
    y_min = random.choice(np.arange(y_min-10,y_min+11))
    y_max = random.choice(np.arange(y_max-10,y_max+11))

    return x_min, x_max, y_min, y_max

def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred
