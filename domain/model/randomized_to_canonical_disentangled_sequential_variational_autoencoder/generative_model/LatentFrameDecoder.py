import copy
import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List
from custom.layer.LinearUnit import LinearUnit
from custom.utility.reparameterize import reparameterize


class LatentFrameDecoder(nn.Module):
    def __init__(self,
                 in_dim      : int,
                 fc_out_dims : List[int],
                 **kwargs) -> None:
        super().__init__()
        self.init_in_dim      = copy.deepcopy(in_dim)
        self.latent_frame_dim = fc_out_dims[-1]
        self.fc_common_dim    = fc_out_dims[-2]

        # ------------ Linear ------------
        modules = nn.ModuleList()
        for out_dim in fc_out_dims[:-1]:
            modules.append(
                LinearUnit(
                    in_dim,
                    out_dim
                )
            )
            in_dim = out_dim

        self.fc_common         = nn.Sequential(*modules)
        self.summary_fc_common = torchinfo.summary(self.fc_common, input_size=(1, self.init_in_dim))

        # ------------ last layer (mean var) ------------
        self.latent_frame_mean   = LinearUnit(in_dim, self.latent_frame_dim)
        self.latent_frame_logvar = LinearUnit(in_dim, self.latent_frame_dim)



    def forward(self,  x: Tensor) -> List[Tensor]:
        num_batch, step, dim = x.shape
        x      = x.view(-1, dim)                  # [num_batch * step, dim]
        x      = self.fc_common(x)                # [num_baelf.fc_common_dim]
        mean   = self.latent_frame_mean(x)        # [num_batch * step, self.latent_frame_dim]
        logvar = self.latent_frame_logvar(x)      # [num_batch * step, self.latent_frame_dim]
        mean   = mean.view(num_batch, step, -1)   # [num_batch, step, self.latent_frame_dim]
        logvar = logvar.view(num_batch, step, -1) # [num_batch, step, self.latent_frame_dim]
        a_sample = reparameterize(mean, logvar) # [num_batch * step, self.latent_frame_dim]
        x = x.view(num_batch, step, x.shape[-1])
        # import ipdb; ipdb.set_trace()
        return mean, logvar, a_sample
