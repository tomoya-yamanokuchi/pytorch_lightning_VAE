import torch
from torch import nn
from torch import Tensor
from typing import List
from torch.nn import functional as F
from omegaconf.omegaconf import OmegaConf

from .inference_model.frame_encoder.FrameEncoderFactory import FrameEncoderFactory
from .inference_model.ContextEncoder import ContextEncoder
from .inference_model.motion_encoder.MotionEncoderFactory import MotionEncoderFactory
from .inference_model.BiLSTMEncoder import BiLSTMEncoder
from .generative_model.frame_decoder.FrameDecoderFactory import FrameDecoderFactory
from .generative_model.MotionPrior import MotionPrior
from .generative_model.ContextPrior import ContextPrior
from .loss.ContrastiveLoss import ContrastiveLoss
from .loss.MutualInformation import MutualInformation


class ContrastiveDisentangledSequentialVariationalAutoencoder(nn.Module):
    def __init__(self,
                 network    : OmegaConf,
                 loss       : OmegaConf,
                 num_train  : int,
                 **kwargs) -> None:
        super().__init__()
        # inference
        self.frame_encoder      = FrameEncoderFactory().create(**network.frame_encoder)
        self.bi_lstm_encoder    = BiLSTMEncoder(in_dim=network.frame_encoder.dim_frame_feature, **network.bi_lstm_encoder)
        self.context_encoder    = ContextEncoder(lstm_hidden_dim=network.bi_lstm_encoder.hidden_dim, **network.context_encoder)
        self.motion_encoder     = MotionEncoderFactory().create(lstm_hidden_dim=network.bi_lstm_encoder.hidden_dim, **network.motion_encoder)
        # generate
        in_dim_decoder          = network.context_encoder.context_dim + network.motion_encoder.state_dim
        self.frame_decoder      = FrameDecoderFactory().create(**network.frame_decoder, in_dim=in_dim_decoder, out_channels=network.frame_encoder.in_channels)
        # prior
        self.context_prior      = ContextPrior(network.context_encoder.context_dim)
        self.motion_prior       = MotionPrior(**network.motion_prior)
        # loss
        self.weight             = loss.weight
        self.contrastive_loss   = ContrastiveLoss(**loss.contrastive_loss)
        self.mutual_information = MutualInformation(num_train)


    def forward(self, img: Tensor, **kwargs) -> List[Tensor]:
        num_batch, step, channle, width, height = img.shape
        encoded_frame                = self.frame_encoder(img)                    # shape = [num_batch, step, conv_fc_out_dims[-1]]
        bi_lstm_out                  = self.bi_lstm_encoder(encoded_frame)
        # posterior
        f_mean, f_logvar, f_sample   = self.context_encoder(bi_lstm_out)          # both shape = [num_batch, context_dim]
        z_mean, z_logvar, z_sample   = self.motion_encoder(bi_lstm_out)  # both shape = [num_batch, step, state_dim]
        # prior
        f_mean_prior, f_logvar_prior                 = self.context_prior.dist(f_mean)
        z_mean_prior, z_logvar_prior, z_sample_prior = self.motion_prior(z_sample)

        # image reconstruction
        f_sample_expand = f_sample.unsqueeze(1).expand(num_batch, step, -1)
        x_recon         = self.frame_decoder(torch.cat((z_sample, f_sample_expand), dim=2))
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
                        x                        ,
                        batch_idx                ,
                        results_dict             ,
                        results_dict_aug_context ,
                        results_dict_aug_dynamics,
                        **kwargs) -> dict:
        # reconstruction likelihood:
        recon_loss   = F.mse_loss(results_dict["x_recon"], x, reduction='sum')
        # context distribution:
        qf           = self.define_normal_distribution(mean=results_dict["f_mean"], logvar=results_dict["f_logvar"])
        pf           = self.define_normal_distribution(mean=results_dict["f_mean_prior"], logvar=results_dict["f_logvar_prior"])
        kld_context  = self.kl_reverse(q=qf, p=pf, q_sample=results_dict["f_sample"])
        # dynamical state distribution:
        qz           = self.define_normal_distribution(mean=results_dict["z_mean"], logvar=results_dict["z_logvar"])
        pz           = self.define_normal_distribution(mean=results_dict["z_mean_prior"], logvar=results_dict["z_logvar_prior"])
        kld_dynamics = self.kl_reverse(q=qz, p=pz, q_sample=results_dict["z_sample"])
        # mutual information
        num_batch   = x.shape[0]

        # ---------------------------------------
        '''
            ⇩　contrastive_loss　の計算が回るようになったら
                augmentation をちゃんと作って計算する
        '''
        f_mean_aug = results_dict_aug_context["f_mean"]
        z_mean_aug = results_dict_aug_dynamics["z_mean"]
        # ---------------------------------------
        contrastive_loss_fx = self.contrastive_loss(results_dict["f_mean"], f_mean_aug)
        contrastive_loss_zx = self.contrastive_loss(results_dict["z_mean"].view(num_batch, -1), z_mean_aug.view(num_batch, -1))

        mutual_info_fz      = self.mutual_information(
            f_dist = (results_dict["f_mean"], results_dict["f_logvar"], results_dict["f_sample"]),
            z_dist = (results_dict["z_mean"], results_dict["z_logvar"], results_dict["z_sample"]),
        )

        # to be minimized
        loss =    recon_loss \
                + self.weight.kld_context           * kld_context \
                + self.weight.kld_dynamics          * kld_dynamics \
                - self.weight.contrastive_loss_fx   * contrastive_loss_fx \
                - self.weight.contrastive_loss_zx   * contrastive_loss_zx \
                + self.weight.mutual_information_fz * mutual_info_fz

        return {
            'loss'                     : loss,
            'recon_loss'               : recon_loss,
            'kld_context'              : kld_context,
            'kld_dynamics'             : kld_dynamics,
            'contrastive_loss_context' : contrastive_loss_fx,
            'contrastive_loss_dynamics': contrastive_loss_zx,
            'mutual_information_fz'    : mutual_info_fz
        }