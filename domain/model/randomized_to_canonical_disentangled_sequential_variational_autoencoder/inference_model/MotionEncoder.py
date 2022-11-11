import copy
import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List
import numpy as np
from custom.layer.LinearUnit import LinearUnit
from custom.utility.reparameterize import reparameterize



class MotionEncoder(nn.Module):
    def __init__(self,
                 in_dim      : int,
                 hidden_dims : int,
                 state_dim   : int,
                 **kwargs) -> None:
        super().__init__()
        latent_frame_dim = copy.deepcopy(in_dim)
        # ------------ fc layer ------------
        modules = nn.ModuleList()
        in_dim  = latent_frame_dim
        for out_dim in hidden_dims:
            modules.append(LinearUnit(in_dim, out_dim))
            in_dim = out_dim
        modules.append(nn.Linear(in_dim, state_dim))
        self.mean         = nn.Sequential(*modules)
        self.summary_mean = torchinfo.summary(self.mean, input_size=(1, latent_frame_dim))

        # ------------ Linear (logvar) ------------
        modules = nn.ModuleList()
        in_dim  = latent_frame_dim
        for out_dim in hidden_dims:
            modules.append(LinearUnit(in_dim, out_dim))
            in_dim = out_dim
        modules.append(nn.Linear(in_dim, state_dim))
        self.logvar         = nn.Sequential(*modules)
        self.summary_logvar = torchinfo.summary(self.logvar, input_size=(1, latent_frame_dim))

        # print("\n\n\n", self.__class__)
        # self.summary = torchinfo.summary(self.hidden, input_size=(2, _in_dim))


    def forward(self,  x: Tensor) -> List[Tensor]:
        num_batch, step, dim = x.shape
        x      = x.view(-1, x.shape[-1])            # shape = [num_batch * step, conv_fc_out_dims[-1]]
        # x      = self.hidden(x)                     # shape = [num_batch * step, hidden_dim]
        mean   = self.mean(x)                       # shape = [num_batch * step, state_dim]
        logvar = self.logvar(x)                     # shape = [num_batch * step, state_dim]
        mean   = mean.view(num_batch, step, -1)     # shape = [num_batch, step, state_dim]
        logvar = logvar.view(num_batch, step, -1)   # shape = [num_batch, step, state_dim]
        sample = reparameterize(mean, logvar)       # shape = [num_batch, step, state_dim]
        return mean, logvar, sample



