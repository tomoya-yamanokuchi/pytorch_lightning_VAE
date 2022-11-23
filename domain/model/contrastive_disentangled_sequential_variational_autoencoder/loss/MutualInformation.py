import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
compute
       log q(z)
    ~= log 1/(NM) sum_m=1^M q(z|x_m)
     = - log(MN) + logsumexp_m(q(z|x_m))
'''


class MutualInformation(nn.Module):
    def __init__(self, num_train: int):
        super(MutualInformation, self).__init__()
        self.num_train = num_train


    def logsumexp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0),
                                        dim=dim, keepdim=keepdim))
        else:
            raise ValueError('Must specify the dimension.')


    def log_density(self, mean, logvar, sample):
        c         = torch.Tensor([np.log(2 * np.pi)]).type_as(sample.data)
        inv_sigma = torch.exp(-logvar)
        tmp       = (sample - mean) * inv_sigma
        return -0.5 * (tmp * tmp + 2 * logvar + c)


    def forward(self, f_dist, z_dist):
        assert type(f_dist) == tuple
        assert type(z_dist) == tuple
        f_mean, f_logvar, f_sample = f_dist
        z_mean, z_logvar, z_sample = z_dist

        num_batch, step, dim_z = z_mean.shape
        dim_f                  = f_mean.shape[-1]

        _log_qf_tmp = self.log_density(
            mean   = f_mean.unsqueeze(0).repeat(step, 1, 1).view(step, 1, num_batch, dim_f),   # [8, 1, 128, dim_z]
            logvar = f_logvar.unsqueeze(0).repeat(step, 1, 1).view(step, 1, num_batch, dim_f), # [8, 1, 128, dim_z]
            sample = f_sample.unsqueeze(0).repeat(step, 1, 1).view(step, num_batch, 1, dim_f), # [8, 128, 1, dim_z]
        )

        _log_qz_tmp = self.log_density(
            mean   = z_mean.transpose(0, 1).view(step, 1, num_batch, dim_z),   # [8, 1, 128, 32]
            logvar = z_logvar.transpose(0, 1).view(step, 1, num_batch, dim_z), # [8, 1, 128, 32]
            sample = z_sample.transpose(0, 1).view(step, num_batch, 1, dim_z), # [8, 128, 1, 32]
        )

        _log_qfz_tmp = torch.cat((_log_qf_tmp, _log_qz_tmp), dim=3) # [8, 128, 128, dim_f + dim_z]

        logq_f  = self.logsumexp(_log_qf_tmp.sum(3),  dim=2, keepdim=False) - math.log(num_batch * self.num_train) # [8, 128]
        logq_z  = self.logsumexp(_log_qz_tmp.sum(3),  dim=2, keepdim=False) - math.log(num_batch * self.num_train) # [8, 128]
        logq_fz = self.logsumexp(_log_qfz_tmp.sum(3), dim=2, keepdim=False) - math.log(num_batch * self.num_train) # [8, 128]
        # step x num_batc
        # # mi_fz   = F.relu(logq_f + logq_z - logq_fz).mean()
        # mi_fz_origin   = F.relu(logq_fz - logq_f - logq_z).mean()

        Hf    = F.relu(-logq_f).mean()
        Hz    = F.relu(-logq_z).mean()
        Hfz   = F.relu(-logq_fz).mean()
        mi_fz = Hf + Hz - Hfz

        # import ipdb; ipdb.set_trace()
        return mi_fz