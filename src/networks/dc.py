import torch
from torch.nn import *
from torch.fft import *


class RealComplex(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(
            torch.complex(x.select(self.dim, 0),
                          x.select(self.dim, 1)), dim=self.dim)


class ComplexReal(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat([x.real, x.imag], dim=self.dim)


class FFT2C(Module):
    def __init__(self, dim=(-2, -1), norm='ortho', centered=True):
        super().__init__()
        self.dim = dim
        self.norm = norm
        self.centered = centered

    def forward(self, x):
        if self.centered:
            return fftshift(fft2(ifftshift(x, dim=self.dim), dim=self.dim,
                                 norm=self.norm), dim=self.dim)
        else:
            return fft2(x, dim=self.dim, norm=self.norm)


class IFFT2C(Module):
    def __init__(self, dim=(-2, -1), norm='ortho', centered=True):
        super().__init__()
        self.dim = dim
        self.norm = norm
        self.centered = centered

    def forward(self, x):
        if self.centered:
            return fftshift(ifft2(ifftshift(x, dim=self.dim), dim=self.dim,
                                  norm=self.norm), dim=self.dim)
        else:
            return ifft2(x, dim=self.dim, norm=self.norm)


class CoilSplit(Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, im, sens):
        return sens * torch.unsqueeze(im, dim=self.dim)


class CoilCombine(Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, im, sens):
        return torch.sum(torch.conj_physical(sens) * im, dim=self.dim)


class DataConsistency(Module):
    """Data consistency at the end of each cascade.

    mode 'blend' (original): at sampled locations replace the network's
        k-space by lamda * ksub + (1 - lamda) * kcnn, one lamda in [0, 1]
        shared by all cascades.
    mode 'varnet' (E2E-VarNet style soft DC): a learned gradient step on the
        data term, taken at the cascade's input and added to the networks'
        refinement:  k_out = kcnn - eta_i * M (kin - ksub),  eta_i >= 0 per
        cascade (initialised to 1).
    """
    def __init__(self, lamda, mode='blend', num_cascades=1):
        super().__init__()
        self.mode = mode
        if mode == 'blend':
            self.lamda = Parameter(torch.tensor(float(lamda)))
        elif mode == 'varnet':
            self.etas = Parameter(torch.ones(num_cascades))
        else:
            raise ValueError(f'unknown data consistency mode {mode!r}')

    def forward(self, kcnn, mask, ksub, kin=None, i=0):
        if self.mode == 'varnet':
            eta = self.etas[i].clamp_min(0.0)
            return kcnn - eta * mask * (kin - ksub)
        lamda = self.lamda.clamp(0.0, 1.0)
        return torch.where(mask == 1,
                           lamda * ksub + (1 - lamda) * kcnn,
                           kcnn)
