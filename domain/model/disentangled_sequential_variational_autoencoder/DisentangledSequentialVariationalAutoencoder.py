import torch
import torchinfo
from torch import nn
from torch import Tensor
from typing import List

from .inference_model.FrameEncoder import FrameEncoder
from .inference_model.ContextEncoder import ContextEncoder
# from .Decoder import Decoder
from torch.nn import functional as F


class DisentangledSequentialVariationalAutoencoder(nn.Module):
    def __init__(self,
                 in_channels      : int,
                 conv_out_channels: int,
                 conv_fc_out_dims : int,
                 lstm_hidden_dim  : int,
                 context_dim      : int,
                 latent_dim       : List = None,
                 **kwargs) -> None:
        super().__init__()

        self.conv_x  = FrameEncoder(
                 in_channels       = in_channels,
                 conv_out_channels = conv_out_channels,
                 conv_fc_out_dims  = conv_fc_out_dims
        )
        # torchinfo.summary(self.conv_x, input_size=(1, 3, 64, 64))

        self.context = ContextEncoder(
            in_dim          = conv_fc_out_dims[-1],
            lstm_hidden_dim = lstm_hidden_dim,
            context_dim     = context_dim
        )
        # import ipdb; ipdb.set_trace()
        torchinfo.summary(self.context, input_size=(128, 8, conv_fc_out_dims[-1]))

        # self.decoder = Decoder(self.encoder.summary, conv_out_channels, latent_dim)


    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encoder.forward(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decoder(z), input, mu, log_var]


    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons      = args[0]
        input       = args[1]
        mu          = args[2]
        log_var     = args[3]

        kld_weight  = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss    = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {
            'loss': loss,
            'Reconstruction_Loss':recons_loss.detach(),
            'KLD' :-kld_loss.detach(),
            # 'betaKLD' :
        }