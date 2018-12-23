# Implements SAP defense (https://arxiv.org/abs/1803.01442)
# Reference code: https://github.com/Guneet-Dhillon/mxnet/commit/4a512b487c9c5ccd040bc43f3b2b04cea5e5fc9c

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class SAP(nn.Module):
    def __init__(self, rate):
        super(SAP, self).__init__()
        self.rate = rate
        self.buffer = None

    def forward(self, x):
        x_abs = torch.abs(x)
        p = x_abs / torch.sum(x_abs, dim=(1,2,3), keepdim=True)
        _, C, H, W = x.size()
        n_sample = int(C * H * W * self.rate)
        p_keep = 1 - torch.exp(-n_sample * p)
        if self.buffer is None:
            self.buffer = torch.rand_like(p_keep).cuda()
        else:
            self.buffer.resize_as_(p_keep).uniform_()
        keep = self.buffer < p_keep
        return x * keep.float() / (p_keep + 1.0e-8)

