import os
# from pathlib import Path
import glob
import tqdm
import math
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as T
import torchvision.transforms.functional as TF

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import marching_cubes as mcubes
import pymeshlab
import imageio
from rich.console import Console
from torch_ema import ExponentialMovingAverage
import ipyplot
from IPython.display import clear_output

import clip

from packaging import version as pver

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.cuda.amp.autocast(enabled=False)

#output the rays
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    #In Python 3.5 you can overload @ as an operator. It is named as __matmul__, 
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def get_checkerboard(fg, n=8):
    B, _, H, W = fg.shape
    colors = torch.rand(2, B, 3, 1, 1, 1, 1, dtype=fg.dtype, device=fg.device)
    h = H // n
    w = W // n
    bg = torch.ones(B, 3, n, h, n, w, dtype=fg.dtype, device=fg.device) * colors[0]
    bg[:, :, ::2, :, 1::2] = colors[1]
    bg[:, :, 1::2, :, ::2] = colors[1]
    bg = bg.view(B, 3, H, W)
    return bg


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    rgbs = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val, color = query_func(pts) # [S, 1] --> [x, y, z]
                    val = val.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    color = color.reshape(len(xs), len(ys), len(zs), 3).detach().cpu().numpy()
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
                    rgbs[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = color
    return u, rgbs


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u, color = extract_fields(bound_min, bound_max, resolution, query_func)

    print(u.shape, u.max(), u.min())
    print(color.shape, color.max(), color.min())
    
    smoothed_surface = mcubes.smooth(u)
    
    vertices_color, triangles_color = mcubes.marching_cubes_color(u, color, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()
    print(b_max_np.shape, b_min_np.shape)

    vertices_color[:,:3] = vertices_color[:,:3] / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices_color, triangles_color


class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.direction_list = ['front', 'left side', 'back', 'right side', 'top', 'bottom']
        # self.output_dir = Path(self.opt.output_dir)

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        # clip model
        clip_model, clip_preprocess = clip.load(self.opt.clip_model, device=self.device, jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
      
        if self.opt.clip_model =="ViT-L/14@336px":
            crop_size = 336
        else:
            crop_size = 224

        # image augmentation https://pytorch.org/vision/main/transforms.html
        if self.opt.clip_aug:
            self.aug = T.Compose([
            T.RandomResizedCrop(crop_size,scale=(0.7, 1.0)),
            # T.GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 1)),
            T.ColorJitter(brightness=(0.2),contrast=(0.2),saturation=(0.2)),
            # T.ColorJitter(brightness=(0.2),contrast=(0.2),saturation=(0.2),hue=(0.25)),
            # T.Resize((crop_size, crop_size)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.aug = T.Compose([
                T.RandomCrop((self.opt.h - 16, self.opt.w - 16)),
                T.Resize((crop_size, crop_size)),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        
        #the gaussian blur is only used for bg augmentation not for image itself
        self.gaussian_blur = T.GaussianBlur(15, sigma=(0.1, 10))
        # T.RandomAdjustSharpness(1+ramdom.random(), 0.9)

        self.aug_eval = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # text prompt
        ref_text = self.opt.text
        if ref_text is not None:
            if not self.opt.dir_text:
                text = clip.tokenize([ref_text]).to(self.device)
                self.text_z = self.clip_model.encode_text(text)
            else:
                texts = []
                #not distinguish between left and right side
                # for d in ['front', 'left side', 'back', 'right side', 'top', 'bottom']:
                for d in self.direction_list:
                    text = f"The {d} view of {ref_text}"
                    texts.append(text)
                texts = clip.tokenize(texts).to(self.device)
                self.text_z = self.clip_model.encode_text(texts)
            self.text_z = self.text_z / self.text_z.norm(dim=-1, keepdim=True)
        else:
            self.text_z = None
        
        # ref image prompt
        ref_image_path = self.opt.image
        if ref_image_path is not None:
            ref_image = Image.open(ref_image_path)
            ref_image = self.clip_preprocess(ref_image)
            #torch_vis_2d(ref_image, True)
            ref_image = ref_image.unsqueeze(0).to(self.device)
            self.ref_image_z = self.clip_model.encode_image(ref_image)
            self.ref_image_z = self.ref_image_z / self.ref_image_z.norm(dim=-1, keepdim=True)
        else:
            self.ref_image_z = None

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        # self.working_dir = (self.opt.output_dir + "/" + self.workspace +"/")
        if self.opt.output_dir is not None:
            os.makedirs(self.opt.output_dir, exist_ok=True)
            os.chdir(self.opt.output_dir)
            self.current_dir = os.getcwd().replace('\\','/')   
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
        self.log(f"[INFO] Training parameters:  {self.opt}")    
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    def sort_images(self, dirpath, valid_extensions=('jpg','jpeg','png')):
        # get filepaths of all files and dirs in the given dir
        valid_files = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath)]
        # filter out directories, no-extension, and wrong extension files
        valid_files = [f for f in valid_files if '.' in f and \
            f.rsplit('.',1)[-1] in valid_extensions and os.path.isfile(f)]

        if not valid_files:
            raise ValueError("No valid images in %s" % dirpath)
        # valid_files.sort(key=os.path.getmtime,reverse=True)
        valid_files = sorted(valid_files)
        return valid_files

    ### ------------------------------	

    def train_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        B, N = rays_o.shape[:2]
        #the H and W in training is clip h and w
        H, W = data['H'], data['W']

        # currently fix white bg, MUST force all rays!
        outputs = self.model.render(rays_o, rays_d, staged=False, perturb=True, force_all_rays=True, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
        
        # clip loss
        pred_ws = outputs['weights_sum'].reshape(B, 1, H, W)
        mask_ws = outputs['mask'].reshape(B, 1, H, W) # near < far

        # torch_vis_2d(pred_ws[0])
        # torch_vis_2d(mask_ws[0])
        
        # make N copies for different augmentations... (WARN OOM)
        pred_rgb = pred_rgb.unsqueeze(0).repeat(self.opt.aug_copy, 1, 1, 1, 1).view(self.opt.aug_copy * B, 3, H, W)

        # moved random bg composition here.
        # TODO: fft bg...
        # TODO: differnt bg_type for different local copies?
        bg_type = random.random()
        if bg_type > 0.5:
            bg_color = torch.rand_like(pred_rgb) # pixel-wise random.
        else:
            bg_color = get_checkerboard(pred_rgb, 8) # checker board random.

        # random blur bg
        bg_color = self.gaussian_blur(bg_color)

        pred_rgb = pred_rgb + (1 - pred_ws) * bg_color

        # if self.global_step % 160 == 0:
        #     torch_vis_2d(pred_rgb[0])
        #     torch_vis_2d(pred_rgb[1])

        # augmentations (crop, resize, ...)
        pred_rgb = self.aug(pred_rgb)

        image_z = self.clip_model.encode_image(pred_rgb)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features

        # clip loss
        loss_clip = 0

        if self.text_z is not None:
            if self.opt.dir_text:
                dirs = data['dir'] # [B,]
                text_z = self.text_z[dirs]
            else:
                text_z = self.text_z

            loss_clip = loss_clip - (image_z * text_z).sum(-1).mean()

        #Only use image prompt in assigned direction
        if self.ref_image_z is not None:
            if self.opt.image_direction is not None:
                dirs = int(data['dir']) 
                if self.opt.image_direction == self.direction_list[dirs]:
                    # self.log(f"[INFO] Using reference image for {self.opt.image_direction}")
                    loss_clip = loss_clip - (image_z * self.ref_image_z).sum(-1).mean()
                # else: 
                #     loss_clip = loss_clip - (image_z * self.ref_image_z).sum(-1).mean()*0.25
            else:
                loss_clip = loss_clip - (image_z * self.ref_image_z).sum(-1).mean()

        # transmittance loss
        pred_tr = (1 - pred_ws) * mask_ws # [B, 1, H, W], T = 1 - weights_sum
        mean_tr = pred_tr.sum() / mask_ws.sum()

        # exponentially anneal (0.5 --> 0.8 in 500 steps)
        tau_t = np.minimum(self.global_step / self.opt.tau_step, 1.0)
        tau = np.exp(np.log(self.opt.tau_0) * (1 - tau_t) + np.log(self.opt.tau_1) * tau_t)

        #torch_vis_2d(pred_tr[0].reshape(H, W))

        loss_tr = - torch.clamp(mean_tr, max=tau).mean()
        
        # origin loss
        origin_thresh = 0 # self.opt.bound * 0.5 ** 2 # if origin is inside sphere(bound/2), no need to further regularize
        loss_origin = torch.clamp((outputs['origin'] ** 2).sum(), min=origin_thresh)

        loss = loss_clip + 0.5 * loss_tr + loss_origin
        #loss = loss_clip + 0.25 * loss_tr

        #print('[DEBUG]', loss_clip.item(), mean_tr.item(), loss_origin.item())

        return pred_rgb, pred_ws, loss

    def eval_step(self, data):

        
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        # currently fix white bg, MUST force all rays!
        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=True, force_all_rays=True, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
        pred_ws = outputs['weights_sum'].reshape(B, 1, H, W)
        mask_ws = outputs['mask'].reshape(B, 1, H, W) # near < far
        
        pred_rgb = pred_rgb + (1 - pred_ws) # simple white bg
        pred_rgb_aug = self.aug_eval(pred_rgb)

        # clip loss
        loss_clip = 0

        image_z = self.clip_model.encode_image(pred_rgb_aug)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True) # normalize features

        if self.text_z is not None:
            if self.opt.dir_text:
                dirs = data['dir'] # [B,]
                text_z = self.text_z[dirs]
            else:
                text_z = self.text_z

            loss_clip = loss_clip - (image_z * text_z).sum(-1).mean()

        #Only use image prompt in assigned direction
        if self.ref_image_z is not None:
            if self.opt.image_direction is not None:
                dirs = int(data['dir']) 
                if self.opt.image_direction == self.direction_list[dirs]:
                    loss_clip = loss_clip - (image_z * self.ref_image_z).sum(-1).mean()
                # else: 
                #     loss_clip = loss_clip - (image_z * self.ref_image_z).sum(-1).mean()*0.25
            else:
                loss_clip = loss_clip - (image_z * self.ref_image_z).sum(-1).mean()

        # transmittance loss
        pred_tr = (1 - pred_ws) * mask_ws # [B, 1, H, W], T = 1 - weights_sum
        mean_tr = pred_tr.sum() / mask_ws.sum()

        # exponentially anneal (0.5 --> 0.8 in 500 steps)
        tau_t = np.minimum(self.global_step / self.opt.tau_step, 1.0)
        tau = np.exp(np.log(self.opt.tau_0) * (1 - tau_t) + np.log(self.opt.tau_1) * tau_t)

        #torch_vis_2d(pred_tr[0].reshape(H, W))

        loss_tr = - torch.clamp(mean_tr, max=tau).mean()

        # currently hard to calc origin when eval...

        loss = loss_clip + 0.5 * loss_tr

        return pred_rgb, pred_ws, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):  
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']


        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        else:
            bg_color = torch.ones(3, device=rays_o.device) # [3]

        # currently fix white bg, MUST force all rays!
        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=perturb, force_all_rays=True, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_ws = outputs['weights_sum'].reshape(B, H, W, 1)

        pred_rgb = pred_rgb + (1 - pred_ws) * bg_color

        pred_ws = pred_ws.reshape(B, 1, H, W)

        return pred_rgb, pred_ws


    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.obj')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.log(f"==> Saving mesh")

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    density_map = self.model.density(pts.to(self.device))
                    sigma = density_map['sigma']
                    color = density_map['color']
            return sigma, color

        def obj2ply(obj_path): 
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(obj_path)
            # #simplify_mesh 
            # ms.meshing_decimation_quadric_edge_collapse(targetfacenum=20000)
            # ms.meshing_decimation_edge_collapse_for_marching_cube_meshes()
            ply_path = os.path.splitext(obj_path)[0] + '.ply'
            self.log(f' ==> saving ply') 
            ms.save_current_mesh(ply_path)
            self.log (f' ==> saved ply to  {self.current_dir}/{ply_path}') 

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        self.log(f"==> Saving obj")
        mcubes.export_obj(vertices, triangles, save_path)
        self.log(f"==> Saved obj to {self.current_dir}/{save_path}")
        for i in range(6):
            #unit is kb
            if os.path.getsize(save_path) > 100:
                break
            time.sleep(5)
        obj2ply(save_path)
        self.log(f"==> Finished saving mesh to {self.current_dir}/{save_path}")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs, mesh_res, mesh_trh):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                #clean output in colab to avoid collapse
                if self.opt.colab and self.epoch % (self.eval_interval * 3) == 0:
                     clear_output()
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)
                self.save_mesh(resolution=mesh_res, threshold=mesh_trh)

               

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'images')

        os.makedirs(save_path, exist_ok=True)
        self.log(f"==> Start Test, save results to  {self.current_dir}/{save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        all_preds = []
        all_preds_depth = []

        with torch.no_grad():

            # update grid
            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            for i, data in enumerate(loader):
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)                
                
                # # if self.opt.color_space == 'linear':
                # preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                all_preds.append(pred)
                cv2.imwrite(os.path.join(save_path,f'{self.name}_{self.epoch}_{i:04d}_rgb.jpg'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                if self.opt.save_depth:
                    pred_depth = preds_depth[0].detach().cpu().numpy()[0]
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    all_preds_depth.append(pred_depth)
                    cv2.imwrite(os.path.join(save_path,f'{self.name}_{self.epoch}_{i:04d}_depth.jpg'), pred_depth)
                pbar.update(loader.batch_size)

        if write_video:
            save_path_video = os.path.join(self.workspace, 'videos')
            os.makedirs(save_path_video, exist_ok=True)
            all_preds = np.stack(all_preds, axis=0)
            imageio.mimwrite(os.path.join(save_path_video,f'{self.name}_{self.epoch}_rgb.mp4'), all_preds, fps=20, quality=8, macro_block_size=1)
            if self.opt.save_depth:
                all_preds_depth = np.stack(all_preds_depth, axis=0)
                imageio.mimwrite(os.path.join(save_path_video,f'{self.name}_{self.epoch}_depth.mp4'), all_preds_depth, fps=20, quality=8, macro_block_size=1)
        
        print(f"display images in   {self.current_dir}/{save_path}")

        if self.opt.colab:
            test_image_display = random.sample(self.sort_images(save_path), 12)
            ipyplot.plot_images(test_image_display, max_images=12, img_width=300,force_b64=True,show_url=False)
            self.log(f"only show random 12 samples in colab. The rest out images could be found under project folder.")
        self.log(f"==> Finished Test.")
    
    # [GUI] train text step.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        
        loader = iter(train_loader)

        for _ in range(step):
            
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % 16 == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_ws, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    
    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):
        
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        
        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth, size=(H, W), mode='nearest').squeeze(1)

        outputs = {
            'image': preds[0].detach().cpu().numpy(),
            'depth': preds_depth[0].detach().cpu().numpy(),
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # update grid
        if self.model.cuda_ray:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_ws, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                # if self.report_metric_at_train:
                #     for metric in self.metrics:
                #         metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            # update grid
            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val
                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    # save image
                    if self.opt.save_interval_img:
                        save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch:04d}_{self.local_step:04d}.jpg')
                        if self.opt.save_depth:
                            save_path_depth = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch:04d}_{self.local_step:04d}_depth.jpg')
                    else:
                        save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.local_step:04d}.jpg')
                        if self.opt.save_depth:
                            save_path_depth = os.path.join(self.workspace, 'validation', f'{self.name}_{self.local_step:04d}_depth.jpg')
                        #save_path_gt = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch:04d}_{self.local_step:04d}_gt.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    pred_rgb = (preds[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, pred_bgr)
                    if self.opt.save_depth:
                        pred_depth_rgb =  (preds_depth[0].detach().cpu().numpy()[0] * 255).astype(np.uint8)
                        cv2.imwrite(save_path_depth, pred_depth_rgb)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)

            if self.opt.colab:
                save_path = os.path.join(self.workspace, 'validation')
                test_image_display = self.sort_images(save_path)
                ipyplot.plot_images(test_image_display, max_images=12, img_width=300,force_b64=True,show_url=False)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    if int (old_ckpt[-12:-8]) % self.opt.ckpt_save_interval != 0:
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
                
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
