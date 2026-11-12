"""l1-ESPIRiT with temporal total variation (TTV-CS): a non-learned k-t
compressed-sensing baseline (PyTorch, runs on the GPU).

    min_x 1/2 ||E x - y||^2 + lam ||D_t x||_1

E = sampling mask x FFT x ESPIRiT coil maps per frame, D_t the finite
difference between consecutive frames (not periodic). Solved with ADMM
(as BART's pics): split z = D_t x,
    x = argmin 1/2 ||E x - y||^2 + rho/2 ||D_t x - z + u||^2  (n_cg CG steps,
        warm-started)
    z = soft(D_t x + u, lam / rho);  u = u + D_t x - z
The data are scaled so that max |E^H y| = 1, which makes lam and rho
independent of the data scaling; the result is scaled back.

Frame phases: CMRxRecon frames carry random global phase flips, which a
temporal-difference penalty would read as large jumps. `phases` (unit
complex per frame, the same estimate as the coil maps, e.g.
utils.vista.frame_phases) are removed from the data first.
"""
import torch


def _fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(
        x, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))


def _ifft2c(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(
        x, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))


def _dt(x):
    return x[1:] - x[:-1]


def _dth(z):
    """Adjoint of _dt: (T-1, ...) -> (T, ...)."""
    pad = torch.zeros_like(z[:1])
    return torch.cat([pad, z]) - torch.cat([z, pad])


def _soft(x, thr):
    mag = x.abs()
    return x * (torch.clamp(mag - thr, min=0) / mag.clamp_min(1e-30))


def ttv_recon(ksub, mask, sens, lam=0.01, rho=1.0, n_admm=50, n_cg=5,
              phases=None):
    """ksub, mask, sens: (T, C, H, W) (mask real). Returns the complex image
    series (T, H, W)."""
    mask = mask.to(ksub.real.dtype)
    y = ksub * mask
    if phases is not None:
        y = y * phases.conj().view(-1, 1, 1, 1)
    E = lambda x: mask * _fft2c(sens * x[:, None])
    EH = lambda k: (sens.conj() * _ifft2c(mask * k)).sum(1)
    ehy = EH(y)
    scale = ehy.abs().max().clamp_min(1e-30)
    ehy = ehy / scale
    y = y / scale
    A = lambda x: EH(E(x)) + rho * _dth(_dt(x))
    x = ehy.clone()
    z = _dt(x)
    u = torch.zeros_like(z)
    for _ in range(n_admm):
        b = ehy + rho * _dth(z - u)
        r = b - A(x)
        p = r.clone()
        rs = torch.vdot(r.flatten(), r.flatten()).real
        for _ in range(n_cg):
            ap = A(p)
            alpha = rs / torch.vdot(p.flatten(), ap.flatten()).real.clamp_min(1e-30)
            x = x + alpha * p
            r = r - alpha * ap
            rs_new = torch.vdot(r.flatten(), r.flatten()).real
            p = r + (rs_new / rs.clamp_min(1e-30)) * p
            rs = rs_new
        dx = _dt(x)
        z = _soft(dx + u, lam / rho)
        u = u + dx - z
    return x * scale
