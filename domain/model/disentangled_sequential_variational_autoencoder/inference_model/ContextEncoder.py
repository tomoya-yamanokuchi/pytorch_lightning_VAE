import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List
import numpy as np
from custom.layer.LinearUnit import LinearUnit
from custom.utility.reparameterize import reparameterize

class ContextEncoder(nn.Module):
    def __init__(self,
                 in_dim           : int,
                 lstm_hidden_dim  : List[int],
                 context_dim      : int,
                 **kwargs) -> None:
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        # ------------ LSTM ------------
        self.lstm_out = nn.LSTM(
            input_size    = in_dim,
            hidden_size   = lstm_hidden_dim,
            num_layers    = 1,
            bidirectional = True, # if True:  output dim is lstm_hidden_dim * 2
            batch_first   = True,
        )
        self.summary = torchinfo.summary(self.lstm_out, input_size=(131, in_dim))
        # ------------ fc layer ------------
        self.context_mean   = LinearUnit(lstm_hidden_dim*2, context_dim)
        self.context_logvar = LinearUnit(lstm_hidden_dim*2, context_dim)
        return


    def forward(self,  x: Tensor) -> List[Tensor]:
        '''
        num_batch, step, dim = lstm_out.shape
        The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        '''
        # num_batch, step, dim  = x.shape
        lstm_out, _             = self.lstm_out(x)                                      # shape=[num_batch, step, lstm_hidden_dim*2]
        forward_last_timestep   = lstm_out[:, -1, :self.lstm_hidden_dim]                # shape=[num_batch, lstm_hidden_dim]
        backward_first_timestep = lstm_out[:,  0, self.lstm_hidden_dim:]                # shape=[num_batch, lstm_hidden_dim]
        assert forward_last_timestep.shape == backward_first_timestep.shape             # check shape consistence
        # import ipdb; ipdb.set_trace()
        lstm_out = torch.cat((forward_last_timestep, backward_first_timestep), dim=1)   # shape=[num_batch, lstm_hidden_dim*2]
        # import ipdb; ipdb.set_trace()
        mean     = self.context_mean(lstm_out)                                          # shape=[num_batch, context_dim]
        logvar   = self.context_logvar(lstm_out)                                        # shape=[num_batch, context_dim]
        sample   = reparameterize(mean, logvar)
        return mean, logvar, sample