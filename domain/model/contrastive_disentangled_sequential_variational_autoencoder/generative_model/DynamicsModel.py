import torch
from torch import Tensor
from torch import nn
from typing import List, Any
from custom import reparameterize


class DynamicsModel(nn.Module):
    def __init__(self,
                 state_dim  : int,
                 hidden_dim : int,
                 **kwargs) -> None:
        super().__init__()
        self.state_dim  = state_dim
        self.hidden_dim = hidden_dim
        self.z_lstm = nn.LSTMCell(
            input_size    = state_dim,
            hidden_size   = hidden_dim,
        )
        self.z_mean   = nn.Linear(hidden_dim, state_dim)
        self.z_logvar = nn.Linear(hidden_dim, state_dim)



    def forward(self, num_batch: int, step: int, device) -> Tensor:
        '''
        xは device（cpu/gpu）を指定するために使用する
        '''
        # initialize initial state z0 (=0)
        z_t        = torch.zeros(num_batch, self.state_dim, device=device)
        z_mean_t   = torch.zeros(num_batch, self.state_dim, device=device)
        z_logvar_t = torch.zeros(num_batch, self.state_dim, device=device)
        h_t        = torch.zeros(num_batch, self.hidden_dim, device=device)
        c_t        = torch.zeros(num_batch, self.hidden_dim, device=device)
        # initialize output tensor for return
        z_means    = torch.zeros(num_batch, step, self.state_dim, device=device)
        z_logvars  = torch.zeros(num_batch, step, self.state_dim, device=device)

        for t in range(step):
            h_t, c_t           = self.z_lstm(z_t, (h_t, c_t))
            z_mean_t           = self.z_mean(h_t)
            z_logvar_t         = self.z_logvar(h_t)
            z_means  [:, t, :] = z_mean_t
            z_logvars[:, t, :] = z_logvar_t

        return z_means, z_logvars
