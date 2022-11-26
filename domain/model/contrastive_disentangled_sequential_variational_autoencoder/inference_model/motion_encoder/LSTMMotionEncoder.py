import torch
import torchinfo
from torch import Tensor
from torch import nn
from typing import List
import numpy as np
from custom.layer.LinearUnit import LinearUnit
from custom.utility.reparameterize import reparameterize


class LSTMMotionEncoder(nn.Module):
    def __init__(self,
                 lstm_hidden_dim  : int,
                 state_dim        : int,
                 **kwargs) -> None:
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim

        # self.rnn    = nn.RNN(lstm_hidden_dim * 2, lstm_hidden_dim, batch_first=True)
        self.lstm = nn.LSTM(
            input_size    = lstm_hidden_dim,
            hidden_size   = lstm_hidden_dim,
            num_layers    = 1,
            bidirectional = False, # if True:  output dim is lstm_hidden_dim * 2
            batch_first   = True,
        )

        # Each timestep is for each z so no reshaping and feature mixing
        self.mean   = nn.Linear(lstm_hidden_dim, state_dim)
        self.logvar = nn.Linear(lstm_hidden_dim, state_dim)

        # self.forward(Tensor(np.random.randn(32, 8, 512)))
        # import ipdb; ipdb.set_trace()


    def forward(self,  bi_lstm_out: Tensor) -> List[Tensor]:
        # num_batch, step, dim = lstm_out.shape
        frontal     = bi_lstm_out[:, :, :self.lstm_hidden_dim] # shape=[num_batch, lstm_hidden_dim]
        features, _ = self.lstm(frontal)           # shape = [num_batch, step, lstm_hidden_dim]
        mean        = self.mean(features)          # shape = [num_batch, step, state_dim]
        logvar      = self.logvar(features)        # shape = [num_batch, step, state_dim]
        sample      = reparameterize(mean, logvar) # shape = [num_batch, step, state_dim]
        return mean, logvar, sample



