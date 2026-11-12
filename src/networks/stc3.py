"""stC^3-Net v2: the (2+1)D complex cross-domain cascade with a physically
consistent signal model.

Differences from MultiCoilComplexTemporalCascadeCrossDomainNet (v1, --stc3):
  * State: one complex image sequence x (B, T, H, W), as in DL-ESPIRiT and
    CineVN, instead of multi-coil k-space that mixes the coil-map model
    S x with measured samples that do not fit it.
  * Data consistency with the ESPIRiT forward model A = M F S, after the
    k-net and after the image net of every cascade:
        'grad': x <- x - eta_j A^H (A x - y)                 (learned eta_j)
        'cg':   after the image net, x <- argmin ||A x - y||^2 + mu_i ||x - z||^2
                (10 CG iterations; the step after the k-net stays 'grad').
  * k-net domain:
        'coil':     per-coil k-space F S x with the measured samples put back,
                    coils as channels (v1); its output is projected back
                    with S^H F^-1.
        'combined': k-space of the coil-combined image F x, one channel, so
                    the weights do not depend on the scan's coil ordering.
        'none':     no k-net (image-only control).
        'acs':      as 'coil', plus a scan-specific SPIRiT kernel G
                    calibrated on the ACS lines (spirit_kernel; closed form,
                    not learned): g = G * k, a SPIRiT step on the unsampled
                    points k <- k + alpha_i (1 - M)(g - k) (learned alpha_i),
                    and the k-net sees [k, g].
    knet_pos gives the k-net (coil, acs) its k-space position as 3 extra
    input channels: k_y, k_x in [-1, 1] and the radius sqrt(k_y^2 + k_x^2),
    so it can treat the centre and the periphery differently.
    knet_linear drops normalisation and activations (a learned linear k-t
    interpolation kernel).
  * Global scale: the input is divided by max |A^H y| and the output
    multiplied back (the networks are then not fed arbitrary scales; the
    k-net needs no instance normalisation dominated by the k-space centre).
  * phase_equivariant: modReLU and magnitude instance normalisation instead
    of CReLU / CInstanceNorm3d (which act on real and imaginary parts
    separately, so the output depends on the arbitrary global phase).
    phase_aug: random global phase of the input during training.
  * inet_real (ablation): the image networks are real (2+1)D UNets on the
    real and imaginary parts as 2 channels, width round(sqrt(2) x
    num_filters) so the parameter count matches the complex UNets.
  * Output 'rss': root-sum-of-squares of F^-1(M y + (1 - M) F S x), the
    data-consistent multi-coil k-space combined as the target is; 'mag': |x|.
  * The final complex image is kept in self.last_image (for a complex loss).

Interface as the temporal models in this repo:
    ksub, mask, sens: (B, 1, T, C, kH, kW)  ->  magnitude (B, 1, T, H, W)
"""
import math
import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter
from torch.utils.checkpoint import checkpoint
from .dc import FFT2C, IFFT2C
from .ctorch import CInstanceNorm3d, CReLU
from .unet import CUNet2p1d, RUNet2p1d, make_residual
from .models import ShallowCKNet2p1d
from utils.vista import frame_phases, time_average


class RealUNet2p1dWrapper(Module):
    """Ablation of the complex image network: a real (2+1)D UNet on the real
    and imaginary parts as 2 channels; complex (B, 1, T, H, W) in and out."""
    def __init__(self, net_depth, num_filters):
        super().__init__()
        self.net = RUNet2p1d(2, 2, net_depth, num_filters)

    def forward(self, z):
        x = torch.cat((z.real, z.imag), dim=1)              # (B, 2, T, H, W)
        y = self.net(x)
        return torch.complex(y[:, :1], y[:, 1:])


def real_unet_filters(num_filters):
    """Width of the real UNet with about the parameters of the complex one
    (a complex conv has twice the weights of a real conv of the same width)."""
    return int(round(num_filters * math.sqrt(2)))


class ModReLU(Module):
    """z * relu(|z| + b) / |z| with a learned bias per channel; changes the
    magnitude only, so it commutes with a global phase."""
    def __init__(self, channels):
        super().__init__()
        self.b = Parameter(torch.empty(channels).uniform_(-0.01, 0.01))

    def forward(self, z):
        mag = z.abs()
        b = self.b.view(1, -1, *([1] * (z.dim() - 2)))
        return z * (torch.relu(mag + b) / (mag + 1e-6))


class CMagInstanceNorm3d(Module):
    """(z - mean) / sqrt(mean |z - mean|^2) per sample and channel, times a
    real gain; commutes with a global phase."""
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gain = Parameter(torch.ones(channels))

    def forward(self, z):
        dims = tuple(range(2, z.dim()))
        z = z - z.mean(dim=dims, keepdim=True)
        var = (z.real ** 2 + z.imag ** 2).mean(dim=dims, keepdim=True)
        return z * (self.gain.view(1, -1, *([1] * len(dims)))
                    / torch.sqrt(var + self.eps))


def make_phase_equivariant(net):
    """Swap CInstanceNorm3d -> CMagInstanceNorm3d and CReLU -> ModReLU (with
    the channel count of the preceding normalisation) in every module."""
    for m in net.modules():
        channels = None
        for name, child in list(m.named_children()):
            if isinstance(child, CInstanceNorm3d):
                channels = child.norm_r.num_features
                setattr(m, name, CMagInstanceNorm3d(channels))
            elif isinstance(child, CReLU):
                if channels is None:
                    raise ValueError(f'no normalisation before CReLU in {m}')
                setattr(m, name, ModReLU(channels))
    return net


def acs_rows(mask):
    """Phase-encoding rows sampled in every frame, contiguous around the
    centre (mask: (B, T, C, H, W); the same rows for the whole batch)."""
    full = (mask[:, :, 0, :, 0] > 0).all(dim=1).all(dim=0)      # (H,)
    h = full.shape[0]
    lo = hi = h // 2
    while lo > 0 and full[lo - 1]:
        lo -= 1
    while hi < h - 1 and full[hi + 1]:
        hi += 1
    return lo, hi + 1


def spirit_kernel(y, mask, size=5, reg=1e-2, calib_rows=16, align=True):
    """SPIRiT kernel G (B, C, C, size, size) from the ACS block of y
    (B, T, C, H, W), pooled over frames: each coil's sample is regressed
    on all coils' size x size neighbourhoods except itself (Tikhonov
    reg x mean diagonal of the normal matrix). G * k (conv2d, zero
    padding) predicts k from its neighbours.
    Masks without an ACS block (VISTA, true R): calibrated on the central
    calib_rows lines of the phase-aligned time average instead (as the
    coil maps, utils.vista.timeavg_maps). align=False averages without the
    frame phase alignment (data without CMRxRecon's phase flips, e.g. OCMR,
    as timeavg_maps(align=False))."""
    lo, hi = acs_rows(mask)
    if hi - lo < size:
        lines = mask[:, :, 0, :, 0] > 0                          # (B, T, H)
        phases = frame_phases(y) if align else torch.ones(
            y.shape[1], dtype=y.dtype, device=y.device)
        kav = time_average(y, lines, phases)                     # (B, C, H, W)
        lo = y.shape[-2] // 2 - calib_rows // 2
        y, hi = kav[:, None], lo + calib_rows
    b, t, c, _, w = y.shape
    acs = y[:, :, :, lo:hi].reshape(b * t, c, hi - lo, w)
    p = torch.complex(F.unfold(acs.real, size), F.unfold(acs.imag, size))
    p = p.reshape(b, t, c * size * size, -1).transpose(1, 2).reshape(
        b, c * size * size, -1)                                  # (B, n, N)
    n = c * size * size
    gram = p @ p.conj().transpose(1, 2)                          # (B, n, n)
    lam = reg * gram.diagonal(dim1=1, dim2=2).real.mean(1)
    centre = (size * size) // 2
    kernel = torch.zeros(b, c, n, dtype=y.dtype, device=y.device)
    for j in range(c):
        t_idx = j * size * size + centre
        keep = torch.arange(n, device=y.device) != t_idx
        a = gram[:, keep][:, :, keep] + lam.view(-1, 1, 1) * torch.eye(
            n - 1, device=y.device)
        rhs = gram[:, keep, t_idx]                                # sum src conj(tgt)
        sol = torch.linalg.solve(a, rhs.unsqueeze(-1))[..., 0]   # (B, n-1)
        kernel[:, j, keep] = sol.conj()
    return kernel.reshape(b, c, c, size, size)


def apply_kernel(kernel, k):
    """G * k for k (B, T, C, H, W), one kernel per batch sample."""
    pad = kernel.shape[-1] // 2
    return torch.stack([F.conv2d(k[i], kernel[i], padding=pad)
                        for i in range(k.shape[0])])


class STC3Net(Module):
    def __init__(self, cascade_depth=8, net_depth=3, num_filters=16,
                 knet_width=16, knet_layers=5, knet_t_kernel=3,
                 knet_domain='coil', knet_linear=False, knet_pos=False,
                 num_coils=10,
                 dc='grad', cg_iter=10, output='rss', phase_equivariant=False,
                 phase_aug=False, grad_checkpoint=False, inet_real=False):
        super().__init__()
        if knet_domain not in ('coil', 'combined', 'none', 'acs'):
            raise ValueError(f'unknown knet_domain {knet_domain!r}')
        if dc not in ('grad', 'cg') or output not in ('rss', 'mag'):
            raise ValueError(f'unknown dc {dc!r} or output {output!r}')
        if inet_real and phase_equivariant:
            raise ValueError('inet_real is not phase-equivariant')
        self.inets = ModuleList([
            RealUNet2p1dWrapper(net_depth, real_unet_filters(num_filters))
            if inet_real else CUNet2p1d(1, 1, net_depth, num_filters)
            for _ in range(cascade_depth)])
        c = 1 if knet_domain == 'combined' else num_coils
        c_in = 2 * c if knet_domain == 'acs' else c
        if knet_pos and knet_domain not in ('coil', 'acs'):
            raise ValueError('knet_pos needs knet_domain coil or acs')
        c_in += 3 if knet_pos else 0
        self.knet_pos = knet_pos
        self.knets = ModuleList([
            ShallowCKNet2p1d(c_in, c, knet_width, knet_layers, knet_t_kernel,
                             linear=knet_linear)
            for _ in range(cascade_depth)]) if knet_domain != 'none' else None
        make_residual(self)
        if phase_equivariant:
            make_phase_equivariant(self)
        # DC after the k-net (even j) and after the image net (odd j)
        self.etas = Parameter(torch.ones(2 * cascade_depth))
        self.mus = Parameter(torch.full((cascade_depth,), 0.05)) \
            if dc == 'cg' else None
        self.alphas = Parameter(torch.full((cascade_depth,), 0.5)) \
            if knet_domain == 'acs' else None
        self.kernel = None                                # set per forward
        self.knet_domain, self.dc, self.cg_iter = knet_domain, dc, cg_iter
        self.output, self.phase_aug = output, phase_aug
        self.grad_checkpoint = grad_checkpoint
        self.fft2c, self.ifft2c = FFT2C(), IFFT2C()
        self.last_image = None

    # ESPIRiT model on k (B, T, C, H, W) and images (B, T, H, W)
    def _FS(self, x, sens):
        return self.fft2c(sens * x.unsqueeze(2))

    def _SHFH(self, k, sens):
        return (torch.conj(sens) * self.ifft2c(k)).sum(dim=2)

    def _grad_dc(self, x, y, mask, sens, j):
        eta = self.etas[j].clamp_min(0.0)
        return x - eta * self._SHFH(mask * self._FS(x, sens) - y, sens)

    def _cg_dc(self, z, y, mask, sens, i):
        mu = self.mus[i].clamp_min(1e-4)

        def normal(p):                                    # (A^H A + mu I) p
            return self._SHFH(mask * self._FS(p, sens), sens) + mu * p

        def dot(a, b):                                    # per sample, real
            return (a.conj() * b).real.flatten(1).sum(1).view(-1, 1, 1, 1)

        rhs = self._SHFH(y, sens) + mu * z
        x, r = z, rhs - normal(z)                         # warm start at z
        p, rs = r, dot(r, r)
        for _ in range(self.cg_iter):
            ap = normal(p)
            alpha = rs / dot(p, ap).clamp_min(1e-30)
            x = x + alpha * p
            r = r - alpha * ap
            rs_new = dot(r, r)
            p = r + rs_new / rs.clamp_min(1e-30) * p
            rs = rs_new
        return x

    def _with_pos(self, z):
        """Append k_y, k_x, radius channels to z (B, C, T, H, W)."""
        if not self.knet_pos:
            return z
        b, _, t, h, w = z.shape
        ky = torch.linspace(-1, 1, h, device=z.device).view(h, 1).expand(h, w)
        kx = torch.linspace(-1, 1, w, device=z.device).view(1, w).expand(h, w)
        pos = torch.stack([ky, kx, torch.sqrt(ky ** 2 + kx ** 2)])
        pos = pos.view(1, 3, 1, h, w).expand(b, 3, t, h, w).to(z.dtype)
        return torch.cat([z, pos], dim=1)

    def _kstep(self, x, y, mask, sens, i):
        knet = self.knets[i]
        if self.knet_domain == 'coil':
            k = self._FS(x, sens)
            k = mask * y + (1 - mask) * k                 # measured samples
            k = knet(self._with_pos(k.transpose(1, 2))).transpose(1, 2)
            return self._SHFH(k, sens)
        if self.knet_domain == 'acs':
            k = self._FS(x, sens)
            k = mask * y + (1 - mask) * k
            g = apply_kernel(self.kernel, k)              # calibrated estimate
            k = k + self.alphas[i] * (1 - mask) * (g - k)  # SPIRiT step
            k = knet(self._with_pos(torch.cat([k, g], dim=2).transpose(1, 2))
                     ).transpose(1, 2)
            return self._SHFH(k, sens)
        z = self.fft2c(x).unsqueeze(1)                    # (B, 1, T, H, W)
        return self.ifft2c(knet(z)[:, 0])

    def _cascade(self, x, y, mask, sens, i):
        if self.knets is not None:
            x = self._kstep(x, y, mask, sens, i)
            x = self._grad_dc(x, y, mask, sens, 2 * i)
        x = self.inets[i](x.unsqueeze(1))[:, 0]
        if self.dc == 'cg':
            return self._cg_dc(x, y, mask, sens, i)
        return self._grad_dc(x, y, mask, sens, 2 * i + 1)

    def forward(self, *args):
        ksub, mask, sens = args[0][:, 0], args[1][:, 0], args[2][:, 0]
        mask = mask.to(ksub.real.dtype)
        x = self._SHFH(ksub, sens)                        # (B, T, H, W)
        s = x.abs().flatten(1).amax(1).clamp_min(1e-12).view(-1, 1, 1, 1)
        y, x = ksub / s.unsqueeze(-1), x / s
        if self.knet_domain == 'acs':
            with torch.no_grad():
                self.kernel = spirit_kernel(
                    y, mask, align=getattr(self, 'calib_align', True))
        phi = None
        if self.phase_aug and self.training:
            phi = torch.exp(1j * 2 * math.pi * torch.rand(
                x.shape[0], 1, 1, 1, device=x.device))
            y, x = y * phi.unsqueeze(-1), x * phi
        for i in range(len(self.inets)):
            if self.grad_checkpoint and torch.is_grad_enabled():
                x = checkpoint(self._cascade, x, y, mask, sens, i,
                               use_reentrant=False)
            else:
                x = self._cascade(x, y, mask, sens, i)
        if self.output == 'rss':
            k = mask * y + (1 - mask) * self._FS(x, sens)
            out = torch.linalg.vector_norm(self.ifft2c(k), dim=2)
        else:
            out = x.abs()
        self.last_image = (x if phi is None else x * phi.conj()) * s
        return (out * s).unsqueeze(1)                     # (B, 1, T, H, W)
