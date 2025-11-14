import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils.logger import *
from torch import Tensor
import time
import sys
import numpy as np
import random
import numpy as np

from models._3DDM.autoencoder import *
from datasets.io import IO

def save_ply_location(i, pc, points_num, name, location):
    if not os.path.exists(location):
        os.makedirs(location)

    L_new = pc.squeeze()
    b = np.float32(L_new) 

    filename = location + str(i) + name
    b = np.hstack([b.reshape(-1, 3)]) 
    np.savetxt(filename, b, fmt='%f %f %f')
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header
'''
    with open(filename, 'r+') as f: 
        old = f.read() 
        f.seek(0) 
        f.write(ply_header % dict(vert_num=len(b))) 
        f.write(old)

def normalize_point_clouds(pcs, mode):
    for i in range(pcs.size(0)):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs, shift, scale

def normalize_partial(pcs, pcs_shift, pcs_scale):
    for i in range(pcs.size(0)):
        pc = pcs[i]
        pc_shift = pcs_shift[i]
        pc_scale = pcs_scale[i]
        pc = (pc - pcs_shift) / pcs_scale
        pcs[i] = pc
    return pcs

def denormalize_pc(pcs, pcs_shift, pcs_scale):
    for i in range(pcs.size(0)):
        pc = pcs[i]
        pc_shift = pcs_shift[i]
        pc_scale = pcs_scale[i]
        pc = (pc * pc_scale) + pc_shift
        pcs[i] = pc
    return pcs

### shapenet 34
def test_shapenet34(args, config):
    logger = get_logger(args.log_name)
    # Model
    logger.info('Loading model...')
    _3DDM = AutoEncoder(args).to(args.gpu)
    # load checkpoints
    builder.load_model(_3DDM, args.ckpts, logger = logger)
    if args.use_gpu:
        _3DDM.to(args.local_rank)
    #  DDP
    if args.distributed:
        raise NotImplementedError()

    network_name = '3DDM_completion_results'
    location_ply = './experiments/demo/' + network_name + '/'
    _3DDM.eval()  # set model to eval mode
    with torch.no_grad():

        gt_root = './experiments/demo/gt/'
        partial_root = './experiments/demo/partial/'
        
        pc_file_list = os.listdir(gt_root)
        for pc_file in pc_file_list:
            model_id = pc_file.split('.')[0]
            gt_file = os.path.join(gt_root, pc_file)
            partial_file = os.path.join(partial_root, pc_file)
            # read single point cloud
            gt_ndarray = IO.get(gt_file).astype(np.float32)
            gt = torch.from_numpy(gt_ndarray).unsqueeze(0).cuda() # [1, 16384, 3]
            partial_ndarray = IO.get(partial_file).astype(np.float32)
            partial = torch.from_numpy(partial_ndarray).unsqueeze(0).cuda() # [1, 2048, 3]

            gt, gt_shift, gt_scale = normalize_point_clouds(gt, 'shape_unit')
            partial = normalize_partial(partial, gt_shift, gt_scale)
            
            code = _3DDM.encode(partial) 
            recons = _3DDM.decode(code, gt.size(1), flexibility=args.flexibility, ret_traj=False) # can be any number

            partial = denormalize_pc(partial, gt_shift, gt_scale)
            recons = denormalize_pc(recons, gt_shift, gt_scale)
            gt = denormalize_pc(gt, gt_shift, gt_scale)
                        
            # #### save all ply files.
            save_ply_location(model_id, recons.cpu().detach().numpy(), recons.size(1), '_3DDM_result.ply', location_ply +'/'  +model_id+'/')
            
    return
