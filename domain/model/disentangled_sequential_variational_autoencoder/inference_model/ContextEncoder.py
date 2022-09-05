import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List
import numpy as np
from custom_network_layer.LinearUnit import LinearUnit


class ContextEncoder(nn.Module):
    def __init__(self,
                 in_dim           : int,
                 lstm_hidden_dim  : List[int],
                 context_dim      : int,
                 **kwargs) -> None:
        super().__init__()
        # ------------ LSTM ------------
        self.lstm_out = nn.LSTM(
            input_size    = in_dim,
            hidden_size   = lstm_hidden_dim,
            num_layers    = 1,
            bidirectional = True, # if True:  output dim is lstm_hidden_dim * 2
            # batch_first   = True
        )
        self.summary = torchinfo.summary(self.lstm_out, input_size=(1, in_dim))
        # ------------ fc layer ------------
        self.context_mean   = nn.Linear(lstm_hidden_dim, context_dim)
        self.context_logvar = nn.Linear(lstm_hidden_dim, context_dim)
        return


    def forward(self,  input: Tensor) -> List[Tensor]:
        """
        - param input: (Tensor) (N, step, dim)
        -      return: (Tensor) []
        """
        lstm_out, _  = self.lstm_out(input)
        lstm_forward = lstm_out[:, :self.lstm_out.hidden_size]
        lstm_reverse = lstm_out[:, self.lstm_out.hidden_size:]
        import ipdb; ipdb.set_trace()
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        # of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        # For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        backward = lstm_out[:,               0, self.hidden_dim:2 * self.hidden_dim]
        frontal  = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)


