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
                 lstm_hidden_dim  : List[int],
                 context_dim      : int,
                 **kwargs) -> None:
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        # self.mean            = LinearUnit(lstm_hidden_dim*2, context_dim, batchnorm=False)
        # self.logvar          = LinearUnit(lstm_hidden_dim*2, context_dim, batchnorm=False)

        self.mean            = nn.Linear(lstm_hidden_dim*2, context_dim)
        self.logvar          = nn.Linear(lstm_hidden_dim*2, context_dim)

        # self.forward(Tensor(np.random.randn(32, 8, lstm_hidden_dim*2)))
        # import ipdb; ipdb.set_trace()


    def forward(self, bi_lstm_out: Tensor) -> List[Tensor]:
        '''
        num_batch, step, dim = lstm_out.shape
        The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        '''
        backward    = bi_lstm_out[:,  0, self.lstm_hidden_dim:] # shape=[num_batch, lstm_hidden_dim]
        frontal     = bi_lstm_out[:, -1, :self.lstm_hidden_dim] # shape=[num_batch, lstm_hidden_dim]
        assert backward.shape == frontal.shape               # check shape consistence
        lstm_out    = torch.cat((frontal, backward), dim=1)  # shape=[num_batch, lstm_hidden_dim*2]
        mean        = self.mean(lstm_out)                    # shape=[num_batch, context_dim]
        logvar      = self.logvar(lstm_out)                  # shape=[num_batch, context_dim]
        sample      = reparameterize(mean, logvar)
        # import ipdb; ipdb.set_trace()
        return mean, logvar, sample