"""VISTA k-t sampling (Variable density Incoherent Spatiotemporal
Acquisition) and calibration from VISTA-undersampled cine data.

vista() is a NumPy port of the reference MATLAB code by Rizwan Ahmad,
github.com/OSU-CMR/VISTA (VISTA.m, fillK.m, dispdup.m, computeU.m; version
1.0, 2015-06-29), with its default parameters (main.m / checkParam.m).
Reference: R. Ahmad, H. Xue, S. Giri, Y. Ding, J. Craft, O.P. Simonetti,
Variable density incoherent spatiotemporal acquisition (VISTA) for highly
accelerated cardiac MRI, Magn Reson Med 2015;74:1266-1278.

    Copyright (c) 2014 - The Ohio State University.
    All rights reserved.

    Permission to use, copy, modify, and distribute this software and its
    documentation for educational, research, and not-for-profit purposes,
    without fee and without written agreement, is hereby granted, provided
    that the above copyright notice, the following two paragraphs, and the
    author attribution appear in all copies of this software. For
    commercial licensing possibilities, contact The Office of Technology
    Commercialization Office (http://tco.osu.edu/) at The Ohio State
    University.

    IN NO EVENT SHALL THE OHIO STATE UNIVERSITY BE LIABLE TO ANY PARTY FOR
    DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING
    OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE OHIO
    STATE UNIVERSITY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    THE OHIO STATE UNIVERSITY SPECIFICALLY DISCLAIMS ANY WARRANTIES
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
    AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS
    ON AN "AS IS" BASIS, AND THE OHIO STATE UNIVERSITY HAS NO OBLIGATION TO
    PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Differences from the MATLAB code: NumPy random numbers (the same seed does
not give the same pattern as MATLAB), no plotting, typ='VISTA' only.
R is the true (net) acceleration: round(p / R) lines in every frame, with no
extra calibration lines.

Calibration (frame_phases, time_average, timeavg_maps; ours, not part of
the reference code): with no ACS lines, coil maps (and the SPIRiT kernel of
stC^3-Net) come from the phase-aligned time average of the undersampled
frames, as in VISTA/GRO cine reconstructions.
"""
import os
import numpy as np


def _round(x):                       # MATLAB round: halves away from zero
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _potential(ph, ti, p_alph, sig, a, s):
    """computeU.m: potential energy of the distribution."""
    scl = 1 - p_alph * np.exp(-ph ** 2 / (2 * sig ** 2))
    d2 = (ph[:, None] - ph[None]) ** 2 + (a * (ti[:, None] - ti[None])) ** 2
    np.fill_diagonal(d2, np.inf)
    return 0.5 * np.sum(np.outer(scl, scl) / d2 ** (s / 2))


def _dispdup(ph, ti, p, t):
    """dispdup.m: move duplicate samples to the nearest vacant phase
    encode of the same frame."""
    ph = (ph + np.ceil((p + 1) / 2)).astype(int)        # 1-based
    ti = (ti + np.ceil((t + 1) / 2)).astype(int)
    pt = (ti - 1) * p + ph
    _, first = np.unique(pt, return_index=True)
    dup = np.setdiff1d(np.arange(pt.size), first)       # all but the first
    dup = dup[np.argsort(pt[dup], kind='stable')]
    empty = np.setdiff1d(np.arange(1, p * t + 1), pt)
    for i in dup:
        x0, y0 = pt[i] - (pt[i] - 1) // p * p, -(-pt[i] // p)
        x, y = empty - (empty - 1) // p * p, -(-empty // p)
        dis = np.where(y == y0, np.abs(x - x0), np.inf)
        b = int(np.argmin(dis))
        pt[i] = empty[b]
        empty = np.delete(empty, b)
    ph = pt - (pt - 1) // p * p - np.ceil((p + 1) / 2)
    ti = -(-pt // p) - np.ceil((t + 1) / 2)
    return ph.astype(float), ti.astype(float)


def _exclude_outer(tmp, p):
    """fillK.m excludeOuter: drop holes contiguous to the k-space edge."""
    tmp = np.sort(tmp)[::-1]
    cnd = np.abs(np.diff(np.concatenate([[np.ceil(p / 2)], tmp])))
    if cnd.size == 0 or cnd.max() == 1:
        return np.array([])
    tmp = tmp[int(np.argmax(cnd > 1)):]
    tmp = np.sort(tmp)
    cnd = np.abs(np.diff(np.concatenate([[-np.floor(p / 2) - 1], tmp])))
    if cnd.size == 0 or cnd.max() == 1:
        return np.array([])
    return tmp[int(np.argmax(cnd > 1)):]


def _center_out(holes):
    order = np.argsort(np.abs(holes), kind='stable')
    return holes[order]


def _fill(ph, ti, p, fr, alph, sig, a, s):
    """fillK.m with fs=1: make the time average fully sampled (except the
    outermost region) by moving samples inward from the boundary side."""
    full = np.arange(-np.floor(fr * p / 2), np.ceil(fr * p / 2))
    holes = _center_out(np.setdiff1d(full, ph))
    while holes.size:
        ind = holes[0]
        sgn = np.sign(ind + np.finfo(float).eps)
        can = ph[sgn * ph > sgn * ind]
        if can.size == 0:
            break
        pcan = can[sgn * (can - ind) == np.min(sgn * (can - ind))][0]
        best, u = None, np.inf
        for j in np.where(ph == pcan)[0]:
            trial = ph.copy()
            trial[j] = ind
            uj = _potential(trial, ti, alph, sig, a, s)
            if uj <= u:
                best, u = j, uj
        ph[best] = ind
        holes = _center_out(_exclude_outer(np.setdiff1d(full, ph), p))
    return ph, ti


def vista(p, t, R, alph=0.28, sig=None, W=None, seed=10, n_iter=120,
          ss=0.25, tf=0.0, s=1.4, fs=1, fr=1.0):
    """VISTA mask (p phase encodes, t frames) at net acceleration R.
    Returns a boolean array (t, p); row t lists frame t's sampled lines."""
    if R == 1:
        return np.ones((t, p), dtype=bool)
    sig = p / 5 if sig is None else sig
    W = max(R / 10 + 0.25, 1) if W is None else W
    g, uni, fl = n_iter // 6, int(_round(n_iter / 2)), int(_round(5 / 6 * n_iter))
    tr = int(_round(p / R))                     # lines per frame
    n = tr * t
    rng = np.random.default_rng(seed)

    # Variable-density random initialisation (VRS)
    p1 = np.arange(-np.floor(p / 2), np.ceil(p / 2))
    prob = 0.1 + alph / (1 - alph + 1e-10) * np.exp(-p1 ** 2 / sig ** 2)
    prob = prob / prob.sum()
    frame_seeds = rng.integers(0, 2 ** 31, t)
    ti = np.repeat(np.arange(-np.floor(t / 2), np.ceil(t / 2)), tr)
    ph = np.concatenate([p1[np.random.default_rng(sd).choice(p, tr, p=prob)]
                         for sd in frame_seeds])

    stp = np.ones(n_iter)
    a = W * np.ones(n_iter)
    dis_ext = np.zeros(n)
    jitter = rng.random(t)
    for i in range(1, n_iter + 1):
        # periodic boundary: 3 x 3 tiling of the k-t plane
        ph = np.concatenate([ph[:n], ph[:n] - p, ph[:n], ph[:n] + p,
                             ph[:n] - p, ph[:n] + p,
                             ph[:n] - p, ph[:n], ph[:n] + p])
        ti = np.concatenate([ti[:n], ti[:n] - t, ti[:n] - t, ti[:n] - t,
                             ti[:n], ti[:n], ti[:n] + t, ti[:n] + t,
                             ti[:n] + t])
        ai = a[i - 1]
        for j in range(n):
            dis = np.sqrt((ph - ph[j]) ** 2 + (ai * (ti - ti[j])) ** 2)
            dis[dis == 0] = np.inf
            scl = 1 - alph * np.exp(-ph ** 2 / (2 * sig ** 2))
            scl = scl + (1 - scl[0])
            dscl = alph * ph[j] * np.exp(-ph[j] ** 2 / (2 * sig ** 2)) / sig ** 2
            fx = s * ((ph[j] - ph) * (scl[j] * scl / dis ** (s + 2))) \
                - dscl * scl / dis ** s
            fy = s * (ai ** 2 * (ti[j] - ti) * (scl[j] * scl / dis ** (s + 2))) * tf
            sfx = stp[i - 1] * fx.sum()
            ph[j] += np.clip(sfx, -R / 4, R / 4)
            ti[j] += np.clip(stp[i - 1] * fy.sum(), -R / 4, R / 4)
            if ph[j] < -np.floor(p / 2) - 0.5:
                ph[j] += p
            elif ph[j] > np.ceil(p / 2) - 0.5:
                ph[j] -= p
            if ti[j] < -np.floor(t / 2) - 0.5:
                ti[j] += t
            elif ti[j] > np.ceil(t / 2) - 0.5:
                ti[j] -= t
            if i % g == 0 or i == n_iter:
                ph[j], ti[j] = _round(ph[j]), _round(ti[j])
            if i == 2:
                dis_ext[j] = abs(sfx)
        if i == 3:
            stp = ss * (1 + R / 4) / np.median(dis_ext) * stp
        ph, ti = ph[:n], ti[:n]
        if i == uni:            # reset to jittered uniform (variable density)
            tmp = np.sort(ph.reshape(t, tr), axis=1)
            ph = np.tile(_round(tmp.mean(0)), t)
            for k in np.arange(-np.floor(t / 2), np.ceil(t / 2)):
                sel = ti == k
                kk = int(k + np.floor(t / 2))
                q = ph[sel] + _round(0.5 * R * (jitter[kk] - 0.5))
                q[q > np.ceil(p / 2) - 1] -= p
                q[q < -np.floor(p / 2)] += p
                ph[sel] = q
            steps = np.arange(i + 1, n_iter + 1)
            a[i:] = a[i:] * (1 + np.exp(-(steps - (i + 1)) / np.ceil(n_iter / 60)))
        if i % g == 0 or i == n_iter:
            ph, ti = _dispdup(ph, ti, p, t)
        if (i % g == 0 or i == n_iter) and i >= fl and fs == 1:
            ph, ti = _fill(ph, ti, p, fr, alph, sig, W, s)
    ph, ti = _dispdup(ph, ti, p, t)
    mask = np.zeros((t, p), dtype=bool)
    mask[(ti + np.floor(t / 2)).astype(int), (ph + np.floor(p / 2)).astype(int)] = True
    return mask


def frame_phases(k, rows=64, eps=0.1, n_iter=3):
    """Global phase of every frame (unit complex, frame 0 -> 1) from k-t
    undersampled k-space k (..., T, C, H, W), zero where not sampled.

    CMRxRecon cine frames carry random global phase flips (e.g. 6 of 12
    frames of a slice rotated by ~180 deg), so the frames must be aligned
    before they are averaged. Spectral phase synchronisation: the pairwise
    phase differences A_ts = <k_t, k_s> over the central `rows` lines come
    from the lines both frames sampled, plus (weight eps) adjacent lines
    with the line-to-line phase ramp of the aligned average removed (needed
    when frames share no line); the phases are the leading eigenvector of
    A, refined n_iter times.
    """
    import torch
    c = k.shape[-2] // 2
    kc = k[..., c - rows // 2:c + rows // 2, :]
    cnt = (kc.abs().sum((-3, -1)) > 0).sum(-2).clamp_min(1)        # (..., rows)
    a0 = torch.einsum('...tcyx,...scyx->...ts', kc.conj(), kc)
    eye = torch.eye(k.shape[-4], dtype=torch.bool, device=k.device)
    ramp = torch.ones(kc.shape[:-4] + (rows - 1,), dtype=k.dtype,
                      device=k.device)
    for _ in range(n_iter):
        b = torch.einsum('...tcyx,...scyx,...y->...ts', kc[..., :-1, :].conj(),
                         kc[..., 1:, :], ramp.conj())
        a = a0 + eps * (b + b.conj().transpose(-1, -2))
        a = a.masked_fill(eye, 0)
        a = a / a.abs().flatten(-2).amax(-1).clamp_min(1e-30)[..., None, None]
        v = torch.linalg.eigh(a.to(torch.complex128))[1][..., -1].to(k.dtype)
        p = v / v.abs().clamp_min(1e-30)
        p = p * p[..., :1].conj()
        kav = (kc * p.conj()[..., None, None, None]).sum(-4) / cnt[..., None, :, None]
        e = torch.einsum('...cyx,...cyx->...y', kav[..., :-1, :].conj(),
                         kav[..., 1:, :])
        ramp = e / e.abs().clamp_min(1e-30)
    return p


def time_average(k, lines, phases):
    """Phase-aligned time average of k (..., T, C, H, W); every line is
    averaged over the frames that sampled it (lines: (..., T, H) bool)."""
    cnt = lines.sum(-2).clamp_min(1).to(k.real.dtype)              # (..., H)
    return (k * phases.conj()[..., None, None, None]).sum(-4) / cnt[..., None, :, None]


def timeavg_maps(ksub, lines, calib_width=16, align=True):
    """ESPIRiT maps (C, H, W) from k-t undersampled k-space ksub (T, C, H, W)
    (torch): frame phases aligned, time-averaged, then the central
    calib_width x calib_width block calibrates ESPIRiT (sigpy, kernel 6,
    no eigenvalue crop, as utils.recon.espirit_map). With 12 frames, a
    16-line block is far more robust than 24 lines: its lines are averaged
    over more frames, so they mix fewer cardiac phases.
    align=False averages without the frame phase alignment: for data whose
    frames have consistent phase (OCMR), where the estimator can invent
    phases once frames share few lines (R >= 8)."""
    import torch
    from sigpy.mri.app import EspiritCalib
    phases = frame_phases(ksub) if align else torch.ones(
        ksub.shape[-4], dtype=ksub.dtype, device=ksub.device)
    kav = time_average(ksub, lines, phases)
    return EspiritCalib(kav.numpy(), calib_width=calib_width, crop=0,
                        max_iter=12, show_pbar=False).run().astype(np.complex64)


_CACHE = {}


def dc_every_frame(lines):
    """VISTA variant with the k-space centre line in every frame: in each
    frame that lacks it, the sampled line farthest from the centre moves
    to the centre (lines per frame, i.e. the true R, unchanged). All frames
    then share a line, so the frame phase estimate (frame_phases) links
    every frame; with plain VISTA the frames split into groups sharing no
    central line at R >= 10-12 (fewer phase encodes: earlier), which broke
    the time-averaged coil maps. lines: (T, H) bool."""
    lines = np.array(lines, dtype=bool, copy=True)
    t, h = lines.shape
    c = h // 2
    for f in range(t):
        if not lines[f, c]:
            on = np.where(lines[f])[0]
            lines[f, on[np.argmax(np.abs(on - c))]] = False
            lines[f, c] = True
    return lines


def vista_mask(p, t, R, seed=10, cache_dir=None, dc=False):
    """vista() with an in-memory and (optional) on-disk cache; dc=True
    returns the dc_every_frame variant of the same pattern."""
    if dc:
        return dc_every_frame(vista_mask(p, t, R, seed, cache_dir))
    key = (p, t, R, seed)
    if key not in _CACHE:
        path = None if cache_dir is None else os.path.join(
            cache_dir, f'vista_p{p}_t{t}_R{R}_sd{seed}.npy')
        if path is not None and os.path.exists(path):
            _CACHE[key] = np.load(path)
        else:
            _CACHE[key] = vista(p, t, R, seed=seed)
            if path is not None:
                os.makedirs(cache_dir, exist_ok=True)
                tmp = f'{path}.{os.getpid()}.tmp.npy'    # parallel jobs
                np.save(tmp, _CACHE[key])
                os.replace(tmp, path)
    return _CACHE[key]
