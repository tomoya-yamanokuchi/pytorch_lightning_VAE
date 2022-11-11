import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List
import numpy as np
import copy
from custom.layer.ConvUnit import ConvUnit
from custom.layer.LinearUnit import LinearUnit
from custom.utility.reparameterize import reparameterize


class FrameEncoder(nn.Module):
    def __init__(self,
                 in_channels      : int,
                 conv_out_channels: List[int],
                 conv_fc_out_dims : List[int],
                 latent_frame_dim : int,
                 **kwargs) -> None:
        super().__init__()

        # ------------ Conv ------------
        modules = nn.ModuleList()
        for out_channels in conv_out_channels:
            modules.append(
                ConvUnit(
                    in_channels,
                    out_channels
                )
            )
            in_channels = out_channels
        self.conv_out         = nn.Sequential(*modules)
        self.summary_conv_out = torchinfo.summary(self.conv_out, input_size=(1, 3, 64, 64))

        # ------------ Linear (mean) ------------
        modules = nn.ModuleList()
        in_dim  = np.prod(self.summary_conv_out.summary_list[-1].output_size)
        _in_dim = copy.deepcopy(in_dim)
        for out_dim in conv_fc_out_dims:
            modules.append(LinearUnit(in_dim, out_dim))
            in_dim = out_dim
        modules.append(nn.Linear(in_dim, latent_frame_dim))
        self.mean         = nn.Sequential(*modules)
        self.summary_mean = torchinfo.summary(self.mean, input_size=(1, _in_dim))

        # ------------ Linear (logvar) ------------
        modules = nn.ModuleList()
        in_dim  = np.prod(self.summary_conv_out.summary_list[-1].output_size)
        _in_dim = copy.deepcopy(in_dim)
        for out_dim in conv_fc_out_dims:
            modules.append(LinearUnit(in_dim, out_dim))
            in_dim = out_dim
        modules.append(nn.Linear(in_dim, latent_frame_dim))
        self.logvar         = nn.Sequential(*modules)
        self.summary_logvar = torchinfo.summary(self.logvar, input_size=(1, _in_dim))



    def forward(self,  x: Tensor) -> List[Tensor]:
        num_batch, step, channle, width, height = x.shape       # 入力データのshapeを取得
        x      = x.view(-1, channle, width, height)             # ;print(x.shape) # 最大で4次元データまでなのでreshapeする必要がある
        x      = self.conv_out(x)                               # ;print(x.shape) # 畳み込みレイヤで特徴量を取得
        x      = torch.flatten(x, start_dim=1)                  # ;print(x.shape) # start_dim 以降の次元を flatten
        mean   = self.mean(x)                                   # ;print(x.shape) # 全結合層で特徴量を抽出
        mean   = mean.view(num_batch, step, mean.shape[-1])     # ;print(x.shape)
        logvar = self.logvar(x)                                 # ;print(x.shape) # 全結合層で特徴量を抽出
        logvar = logvar.view(num_batch, step, logvar.shape[-1]) # ;print(x.shape) # 形状をunrollしてたのを元に戻す(じゃないとLSTMとかに渡せない)
        sample = reparameterize(mean, logvar)
        return mean, logvar, sample


