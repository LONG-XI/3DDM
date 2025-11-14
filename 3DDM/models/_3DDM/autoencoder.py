import torch
from torch.nn import Module
import sys
from .common import *
from .encoders import *
from ._3DDM import *

class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(zdim=args.latent_dim)

        print('\n')
        print('!!!!!Please Note !!!!!')
        print('The main file/codes "_3DDM.py" of our 3DDM will be made publicly available upon the paper’s acceptance and publication.')
        print('The python file "_3DDM.py" contains the main codes for the forward process, reverse process and loss function in our 3DDM')
        print('The trained model file "ckpt-3DDM.pth" will also be made publicly available upon the paper’s acceptance and publication.')
        print('\n')
        sys.exit(0)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )
    
    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code, _ = self.encoder(x) # [B, 256, 1]

        return code

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, gt, partial):
        code = self.encode(partial)
        loss = self.diffusion.get_loss(gt, code)
        return loss
