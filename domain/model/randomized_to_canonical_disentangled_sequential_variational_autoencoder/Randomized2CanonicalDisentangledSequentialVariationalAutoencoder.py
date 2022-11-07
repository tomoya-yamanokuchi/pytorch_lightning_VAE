import torch
from torch import nn
from torch import Tensor
from typing import List
from torch.nn import functional as F
from omegaconf.omegaconf import OmegaConf

from .inference_model.FrameEncoder import FrameEncoder
from .inference_model.ContentEncoder import ContentEncoder
from .inference_model.MotionEncoder import MotionEncoder
from .generative_model.LatentFrameDecoder import LatentFrameDecoder
from .generative_model.FrameDecoder import FrameDecoder
from .generative_model.DynamicsModel import DynamicsModel
from .generative_model.ContentPrior import ContentPrior



class Randomized2CanonicalDisentangledSequentialVariationalAutoencoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 network    : OmegaConf,
                 **kwargs) -> None:
        super().__init__()
        self.frame_encoder        = FrameEncoder(in_channels, **network.frame_encoder)
        self.content_encoder      = ContentEncoder(network.frame_encoder.conv_fc_out_dims[-1], **network.content_encoder)
        self.motion_encoder       = MotionEncoder(network.frame_encoder.conv_fc_out_dims[-1], **network.motion_encoder)
        self.latent_frame_decoder = LatentFrameDecoder(network.content_encoder.content_dim + network.motion_encoder.state_dim, **network.latent_frame_decoder)
        self.frame_decoder        = FrameDecoder(network.latent_frame_decoder.fc_out_dims[-1], **network.frame_decoder)
        self.dynamics_model       = DynamicsModel(**network.dynamics_model)
        self.content_prior        = ContentPrior(network.content_encoder.content_dim)


    def forward(self, img_ran: Tensor, **kwargs) -> List[Tensor]:
        num_batch, step, channle, width, height = img_ran.shape
        a_mean_encoded, a_logvar_encoded, a_sample_encoded  = self.frame_encoder(img_ran) # shape = [num_batch, step, conv_fc_out_dims[-1]]
        # context:
        f_mean, f_logvar, f_sample   = self.content_encoder(a_sample_encoded)          # both shape = [num_batch, context_dim]
        f_mean_prior                 = self.content_prior.mean(f_mean)
        f_logvar_prior               = self.content_prior.logvar(f_logvar)
        # dynamical state:
        z_mean, z_logvar, z_sample   = self.motion_encoder(a_sample_encoded)  # both shape = [num_batch, step, state_dim]
        z_mean_prior, z_logvar_prior = self.dynamics_model(num_batch, step, device=img_ran.device)
        # latent frame decode
        a_mean_decoded, a_logvar_decoded, a_sample_decoded = self.latent_frame_decoder(torch.cat((z_sample, f_sample.unsqueeze(1).expand(num_batch, step, -1)), dim=2))
        # import ipdb; ipdb.set_trace()
        # image reconstruction
        x_recon                      = self.frame_decoder(a_sample_encoded)
        return  {
            "f_mean"          : f_mean,
            "f_logvar"        : f_logvar,
            "f_sample"        : f_sample,
            "f_mean_prior"    : f_mean_prior,
            "f_logvar_prior"  : f_logvar_prior,

            "z_mean"          : z_mean,
            "z_logvar"        : z_logvar,
            "z_sample"        : z_sample,
            "z_mean_prior"    : z_mean_prior,
            "z_logvar_prior"  : z_logvar_prior,

            "a_mean_encoded"  : a_mean_encoded,
            "a_logvar_encoded": a_logvar_encoded,
            "a_sample_encoded": a_sample_encoded,

            "a_mean_decoded"  : a_mean_decoded,
            "a_logvar_decoded": a_logvar_decoded,
            "a_sample_decoded": a_sample_decoded,

            "x_recon"         : x_recon,
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
        num_batch    = kwargs["x_can"].size()[0]
        # reconstruction likelihood:
        recon_loss   = F.mse_loss(kwargs["x_recon"], kwargs["x_can"], reduction='sum') / num_batch
        # Entropy of latent frame
        qa                   = self.define_normal_distribution(kwargs["a_mean_encoded"], kwargs["a_logvar_encoded"])
        latent_frame_entropy = - qa.log_prob(kwargs["a_sample_encoded"]).sum() / num_batch # マイナスの符号に注意
        # Decode latent frame
        pa                   = self.define_normal_distribution(mean=kwargs["a_mean_decoded"], logvar=kwargs["a_logvar_decoded"])
        latent_frame_decode_loglikelihood = pa.log_prob(kwargs["a_sample_encoded"]).sum() / num_batch
        # context distribution:
        qf           = self.define_normal_distribution(mean=kwargs["f_mean"], logvar=kwargs["f_logvar"])
        pf           = self.define_normal_distribution(mean=kwargs["f_mean_prior"], logvar=kwargs["f_logvar_prior"])
        kld_content  = self.kl_reverse(q=qf, p=pf, q_sample=kwargs["f_sample"]) / num_batch
        # dynamical state distribution:
        qz           = self.define_normal_distribution(mean=kwargs["z_mean"], logvar=kwargs["z_logvar"])
        pz           = self.define_normal_distribution(mean=kwargs["z_mean_prior"], logvar=kwargs["z_logvar_prior"])
        kld_motion   = self.kl_reverse(q=qz, p=pz, q_sample=kwargs["z_sample"]) / num_batch

        loss_weight = kwargs["loss_weight"]

        elbo =  - (loss_weight.recon_loss * recon_loss) \
                + (loss_weight.latent_frame_entropy * latent_frame_entropy) \
                + (loss_weight.latent_frame_decode_loglikelihood * latent_frame_decode_loglikelihood) \
                - (loss_weight.kld_content * kld_content) \
                - (loss_weight.kld_motion * kld_motion) # to be maximized

        loss = -elbo # to be minimized

        return {
            'loss'                             : loss,
            'recon_loss'                       : (loss_weight.recon_loss * recon_loss),
            'latent_frame_entropy'             : (loss_weight.latent_frame_entropy * latent_frame_entropy),
            'latent_frame_decode_loglikelihood': (loss_weight.latent_frame_decode_loglikelihood * latent_frame_decode_loglikelihood),
            'kld_content'                      : (loss_weight.kld_content * kld_content),
            'kld_motion'                       : (loss_weight.kld_motion * kld_motion),
        }