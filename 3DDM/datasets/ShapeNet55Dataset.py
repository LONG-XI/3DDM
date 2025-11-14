import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
import logging
import os, sys
from tqdm.auto import tqdm

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

@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            if taxonomy_id == '02691156': # airplane
                # ###for training one single class. Codes for training all 34 categories will be updated once our paper is published.
                self.file_list.append({
                    'taxonomy_id': taxonomy_id,
                    'model_id': model_id,
                    'file_path': line
                })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        subset = self.subset
        data = {}

        path_complete = os.path.join(self.pc_path, subset, 'complete', sample['taxonomy_id'], sample['model_id'] + '.h5')
        data_complete = IO.get(path_complete).astype(np.float32)
        data_complete = torch.from_numpy(data_complete).float()

        path_partial = os.path.join(self.pc_path, subset, 'partial', sample['taxonomy_id'], sample['model_id'] + '.h5')
        data_partial = IO.get(path_partial).astype(np.float32)
        data_partial = torch.from_numpy(data_partial).float()
        
        original_partial = data_partial.clone()
        original_gt = data_complete.clone()

        gt_shape_unit, gt_shift, gt_scale = normalize_point_clouds(data_complete.unsqueeze(0), 'shape_unit')
        partial_shape_unit = normalize_partial(data_partial.unsqueeze(0), gt_shift, gt_scale)
        
        return sample['taxonomy_id'], sample['model_id'], (partial_shape_unit.squeeze(0), gt_shape_unit.squeeze(0)), (gt_shift, gt_scale), (original_partial, original_gt)


    def __len__(self):
        return len(self.file_list)
