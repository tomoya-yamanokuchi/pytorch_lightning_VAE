import torch
from torch import nn
from torch import Tensor
from typing import List
from torch.nn import functional as F
from omegaconf.omegaconf import OmegaConf

from .inference_model.FrameEncoder import FrameEncoder
from .inference_model.ContextEncoder import ContextEncoder
from .inference_model.DynamicalStateEncoder import DynamicalStateEncoder
from .generative_model.FrameDecoder import FrameDecoder
from .generative_model.DynamicsModel import DynamicsModel
from .generative_model.ContextPrior import ContextPrior



class DisentangledSequentialVariationalAutoencoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 network    : OmegaConf,
                 **kwargs) -> None:
        super().__init__()
        self.frame_encoder           = FrameEncoder(in_channels, **network.frame_encoder)
        self.context_encoder         = ContextEncoder(network.frame_encoder.conv_fc_out_dims[-1], **network.context_encoder)
        self.dynamical_state_encoder = DynamicalStateEncoder(network.frame_encoder.conv_fc_out_dims[-1], **network.dynamical_state_encoder)
        self.frame_decoder           = FrameDecoder(network.context_encoder.context_dim + network.dynamical_state_encoder.state_dim, **network.frame_decoder)
        self.dynamics_model          = DynamicsModel(**network.dynamics_model)
        self.context_prior           = ContextPrior(network.context_encoder.context_dim)


    def forward(self, img: Tensor, **kwargs) -> List[Tensor]:
        num_batch, step, channle, width, height = img.shape
        encoded_frame                = self.frame_encoder(img)                    # shape = [num_batch, step, conv_fc_out_dims[-1]]
        # context:
        f_mean, f_logvar, f_sample   = self.context_encoder(encoded_frame)          # both shape = [num_batch, context_dim]
        f_mean_prior                 = self.context_prior.mean(f_mean)
        f_logvar_prior               = self.context_prior.logvar(f_logvar)
        # dynamical state:
        z_mean, z_logvar, z_sample   = self.dynamical_state_encoder(encoded_frame)  # both shape = [num_batch, step, state_dim]
        z_mean_prior, z_logvar_prior = self.dynamics_model(num_batch, step, device=img.device)
        # image reconstruction
        x_recon                      = self.frame_decoder(torch.cat((z_sample, f_sample.unsqueeze(1).expand(num_batch, step, -1)), dim=2))
        return  {
            "f_mean"         : f_mean,
            "f_logvar"       : f_logvar,
            "f_sample"       : f_sample,
            "f_mean_prior"   : f_mean_prior,
            "f_logvar_prior" : f_logvar_prior,
            "z_mean"         : z_mean,
            "z_logvar"       : z_logvar,
            "z_sample"       : z_sample,
            "z_mean_prior"   : z_mean_prior,
            "z_logvar_prior" : z_logvar_prior,
            "x_recon"        : x_recon
        }


    def decode(self, z, f):
        '''
        input:
            - z: shape = []
            - f: shape = []
        '''
        num_batch = 1
        step      = 1
        x_recon   = self.frame_decoder(torch.cat((z, f.unsqueeze(1).expand(num_batch, step, -1)), dim=2))
        return x_recon



    def kl_reverse(self, q, p, q_sample):
        kl = (q.log_prob(q_sample) - p.log_prob(q_sample))
        kl = kl.sum()
        return kl


    def define_normal_distribution(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return torch.distributions.Normal(mean, std)


    def loss_function(self,
                      **kwargs) -> dict:
        # reconstruction likelihood:
        recon_loss   = F.mse_loss(kwargs["x_recon"], kwargs["x"], reduction='sum')
        # context distribution:
        qf           = self.define_normal_distribution(mean=kwargs["f_mean"], logvar=kwargs["f_logvar"])
        pf           = self.define_normal_distribution(mean=kwargs["f_mean_prior"], logvar=kwargs["f_logvar_prior"])
        kld_context  = self.kl_reverse(q=qf, p=pf, q_sample=kwargs["f_sample"])
        # dynamical state distribution:
        qz           = self.define_normal_distribution(mean=kwargs["z_mean"], logvar=kwargs["z_logvar"])
        pz           = self.define_normal_distribution(mean=kwargs["z_mean_prior"], logvar=kwargs["z_logvar_prior"])
        kld_dynamics = self.kl_reverse(q=qz, p=pz, q_sample=kwargs["z_sample"])

        loss = recon_loss + kld_context + kld_dynamics

        return {
            'loss'         : loss,
            'recon_loss'   : recon_loss,
            'kld_context'  : kld_context,
            'kld_dynamics' : kld_dynamics,
        }