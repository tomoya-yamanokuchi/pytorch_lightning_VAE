import torch
from torch import Tensor


def reparameterize(mean: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + (eps * std)