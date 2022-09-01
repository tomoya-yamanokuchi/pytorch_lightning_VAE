import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List
import numpy as np
import copy
from custom_network_layer.ConvUnit import ConvUnit
from custom_network_layer.LinearUnit import LinearUnit

class FrameEncoder(nn.Module):
    def __init__(self,
                 in_channels      : int,
                 conv_out_channels: List[int],
                 conv_fc_out_dims : List[int],
                 **kwargs) -> None:
        super().__init__()

        # ------------ Conv ------------
        modules = []
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

        # ------------ Linear ------------
        modules = []
        in_dim = np.prod(self.summary_conv_out.summary_list[-1].output_size)
        for out_dim in conv_fc_out_dims:
            modules.append(
                LinearUnit(
                    in_dim,
                    out_dim
                )
            )
            in_dim = out_dim
        self.conv_fc = nn.Sequential(*modules)
        self.summary_conv_fc = torchinfo.summary(self.conv_fc, input_size=(1, conv_fc_out_dims[0]))
        return


    def forward(self,  input: Tensor) -> List[Tensor]:
        """
        - param input: (Tensor) Input tensor to encoder [N x C x H x W]
        -      return: (Tensor) List of latent codes
        """
        result        = self.conv_out(input)                ;print(result.shape)
        result        = torch.flatten(result, start_dim=1)  ;print(result.shape)
        encoded_frame = self.conv_fc(result)                ;print(result.shape)
        return encoded_frame


