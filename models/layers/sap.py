# Implements SAP defense (https://arxiv.org/abs/1803.01442)
# Reference code: https://github.com/Guneet-Dhillon/mxnet/commit/4a512b487c9c5ccd040bc43f3b2b04cea5e5fc9c

import torch
import torch.nn as nn

class SAP(nn.Module):
    def __init__(self, frac):
        """Intialization
        frac: fraction of elements to keep (sample with replacement)
        """
        super(SAP, self).__init__()
        self.frac = frac
        self.mask = None

    def forward(self, x):
        """Forward propagation of SAP, sets elements in x to zero randomly"""
        shape = x.size()
        assert len(shape) >= 2
        x = x.view(shape[0], -1)
        keep = int(x.size(1) * self.frac)
        abs_x = torch.abs(x)
        keep_idx = torch.multinomial(abs_x, keep, replacement=True)
        # make an all-ones buffer
        if self.mask is None:
            self.mask = torch.Tensor(x.size()).fill_(1).cuda()
        else:
            self.mask.resize_(x.size()).fill_(1)
        # set some elements to zeros
        self.mask.scatter_(1, keep_idx, 0)
        # fill x with zeros given mask
        x = x * self.mask
        # rescale x with probability
        prob = abs_x / torch.sum(abs_x, dim=1, keepdim=True)
        scale = 1.0 - torch.pow(1.0 - prob, keep)
        scale = torch.clamp(scale, min=1.0e-4)
        # NOTE do we need to detach() *scale*? because *scale* also
        # depends on *x*. In the official MxNet implementation, it use detach()
        out = x / scale.detach()
        return out.view(shape)

if __name__ == "__main__":
    sap1 = SAP(0.5)
    sap2 = SAP(0.5)
    x = torch.randn(2, 30, 18).cuda()
    sap1(x)
    sap2(x)
