import os
import numpy as np
import torch
from torch.fft import *
from utils.vista import vista_mask, timeavg_maps

MASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'masks', 'vista')


class ToTensor(object):
    def __call__(self, item):
        item['kfull'] = torch.from_numpy(item['kfull'])
        if 'sens' in item:
            item['sens'] = torch.from_numpy(item['sens'])
        return item


class CropReadout(object):
    """Remove readout (last axis) oversampling by cropping the centre of the
    image. The readout is fully sampled, so this commutes with the
    phase-encode undersampling applied later. Sensitivity maps are
    image-domain quantities and are cropped directly.
    """
    def __init__(self, factor=2):
        self.factor = factor

    def __call__(self, item):
        w = item['kfull'].shape[-1]
        n = w // self.factor
        start = (w - n) // 2
        im = fftshift(ifft2(ifftshift(item['kfull'], dim=(-2, -1)),
                            dim=(-2, -1), norm='ortho'), dim=(-2, -1))
        im = im[..., start:start + n]
        item['kfull'] = fftshift(fft2(ifftshift(im, dim=(-2, -1)),
                                      dim=(-2, -1), norm='ortho'), dim=(-2, -1))
        if 'sens' in item:
            item['sens'] = item['sens'][..., start:start + n]
        return item


class RandomFlip(object):
    """Random horizontal/vertical flips, applied in image space.

    Flipping centred k-space of even size flips the image AND shifts it by
    one pixel, which would misalign it with the (image-domain) sensitivity
    maps. So the image is flipped instead, together with the maps.
    """
    def __call__(self, item):
        dims = [d for d in (-1, -2) if torch.rand((1,)) < 0.5]
        return flip_item(item, dims)


def flip_item(item, dims):
    """RandomFlip with given dims (-1: readout 'x', -2: phase encode 'y')."""
    item['flip'] = ''.join({-1: 'x', -2: 'y'}[d] for d in dims)
    if not dims:
        return item
    im = fftshift(ifft2(ifftshift(item['kfull'], dim=(-2, -1)),
                        dim=(-2, -1), norm='ortho'), dim=(-2, -1))
    im = torch.flip(im, dims)
    item['kfull'] = fftshift(fft2(ifftshift(im, dim=(-2, -1)),
                                  dim=(-2, -1), norm='ortho'), dim=(-2, -1))
    if 'sens' in item:
        item['sens'] = torch.flip(item['sens'], dims)
    return item


class TargetRecon(object):
    def __call__(self, item):
        item['imfull'] = torch.abs(fftshift(ifft2(ifftshift(
                item['kfull'], dim=(-2, -1)),
                dim=(-2, -1), norm='ortho'),
                dim=(-2, -1)))
        if 'sens' in item or item.get('multi_coil'):
            item['imfull'] = torch.sqrt(torch.sum(
                item['imfull'] ** 2, dim=-3))
        return item


class CenterFrame(object):
    """2.5D: keep only the centre frame of the target (the input keeps all
    2n + 1 frames as context)."""
    def __init__(self, n):
        self.n = n

    def __call__(self, item):
        item['imfull'] = item['imfull'][self.n]
        return item


def vista_map_path(fname, r, h, w, flip='', align=True, pattern='vista'):
    """Cache file of the VISTA coil maps of slice file fname (maps made
    without frame phase alignment get the suffix _noalign, maps of the
    vista_dc pattern the suffix _dc)."""
    name = os.path.basename(fname)[:-3]
    suffix = ('' if align else '_noalign') + \
        ('_dc' if pattern == 'vista_dc' else '')
    return os.path.join(os.path.dirname(fname), '..', 'vista_maps',
                        f'{name}_R{r}_{h}x{w}_flip{flip or "0"}{suffix}.npy')


class Subsampling(object):
    """Phase-encode undersampling at acceleration R (drawn from
    subsample_ratios per sample), always including the central lines.

    pattern 'fixed' (challenge setup): every R-th line, the same in every
        frame.
    pattern 'interleave' (k-t lattice): frame t samples every R-th line
        starting at t mod R, so R consecutive frames cover all lines.
    pattern 'ktgaussian' (k-t variable-density random): each frame draws its
        own extra lines without replacement from a Gaussian density centred
        on k-space (std kt_sigma * kH); exactly as many lines per frame as
        'fixed' at the same R.
    pattern 'vista' (Ahmad et al., MRM 2015; utils/vista.py): one fixed
        VISTA pattern per (kH, number of frames, R), stored in masks/vista/.
        R is the TRUE acceleration (round(kH / R) lines per frame, no extra
        ACS lines; center_lines is ignored). Multi-coil: the coil maps are
        calibrated from the sampled data itself (phase-aligned time
        average, utils.vista.timeavg_maps) instead of being read from the
        file, and cached on disk (float16) per (slice, R, flip) in
        <data dir>/../vista_maps/. Needs all frames of the slice (temporal
        dataset). vista_align=False skips the frame phase alignment (for
        data without CMRxRecon's random per-frame phase flips, e.g. OCMR);
        those maps are cached as *_noalign.npy.
    pattern 'vista_dc': the same VISTA pattern with the k-space centre line
        in every frame (utils.vista.dc_every_frame; same lines per frame),
        so that the frame phase estimate links all frames. Maps cached as
        *_dc.npy.
    Frame indices come from item['frame'] (one frame) or item['frames'] (one
    per leading entry of kfull). seed=None draws fresh random masks
    (training); an integer makes 'ktgaussian' masks deterministic per
    (frame, R) (validation).
    """
    def __init__(self, center_lines, subsample_ratios, pattern='fixed',
                 kt_sigma=0.25, seed=None, vista_align=True):
        if pattern is True:             # old interleave=True
            pattern = 'interleave'
        elif pattern is False or pattern is None:
            pattern = 'fixed'
        if pattern not in ('fixed', 'interleave', 'ktgaussian', 'vista',
                           'vista_dc'):
            raise ValueError(f'unknown sampling pattern {pattern!r}')
        self.center_lines = center_lines
        self.subsample_ratios = subsample_ratios
        self.pattern = pattern
        self.kt_sigma = kt_sigma
        self.seed = seed
        self.vista_align = vista_align

    def _lines(self, h, r, frame):
        """Boolean (kH,) line mask for one frame."""
        lines = torch.zeros(h, dtype=torch.bool)
        c0 = (h - self.center_lines) // 2
        lines[c0:c0 + self.center_lines] = True
        if self.pattern == 'fixed':
            lines[::r] = True
        elif self.pattern == 'interleave':
            lines[int(frame) % r::r] = True
        else:
            n_total = int((lines | (torch.arange(h) % r == 0)).sum())
            cand = torch.where(~lines)[0]
            k = cand.float() - (h - 1) / 2
            w = torch.exp(-0.5 * (k / (self.kt_sigma * h)) ** 2)
            g = None
            if self.seed is not None:
                g = torch.Generator().manual_seed(
                    self.seed * 1000003 + int(frame) * 1009 + r)
            pick = torch.multinomial(w, n_total - int(lines.sum()),
                                     replacement=False, generator=g)
            lines[cand[pick]] = True
        return lines

    def _vista_maps(self, item, r, lines):
        """Time-averaged ESPIRiT maps (T, C, kH, kW), disk-cached."""
        ksub = item['ksub']
        h, w = ksub.shape[-2:]
        path = vista_map_path(item['fname'], r, h, w, item.get('flip', ''),
                              self.vista_align, self.pattern)
        try:
            sens = np.load(path)
            sens = sens[0].astype(np.complex64) + 1j * sens[1].astype(np.complex64)
        except (FileNotFoundError, ValueError, EOFError, OSError):
            sens = timeavg_maps(ksub, lines, align=self.vista_align)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f'{path}.{os.getpid()}.tmp.npy'
            np.save(tmp, np.stack([sens.real, sens.imag]).astype(np.float16))
            os.replace(tmp, path)
        return torch.from_numpy(sens).expand(ksub.shape)

    def __call__(self, item):
        kfull = item['kfull']
        h = kfull.shape[-2]
        r = self.subsample_ratios[
            int(torch.randint(len(self.subsample_ratios), (1,)))]
        mask = torch.zeros_like(kfull, dtype=torch.float)
        if self.pattern in ('vista', 'vista_dc'):
            frames = item.get('frames')
            if frames is None or len(frames) != item['nt']:
                raise ValueError('vista sampling needs all frames of the '
                                 'slice (temporal dataset, no clips)')
            lines = torch.from_numpy(vista_mask(
                h, item['nt'], r, cache_dir=MASK_DIR,
                dc=self.pattern == 'vista_dc'))
            mask[lines.view(lines.shape[0], *([1] * (kfull.dim() - 3)),
                            h, 1).expand_as(mask)] = 1
            item['mask'] = mask
            item['ksub'] = mask * kfull
            if item.get('multi_coil'):
                item['sens'] = self._vista_maps(item, r, lines)
            return item
        if 'frames' in item and self.pattern != 'fixed':
            for j, t in enumerate(item['frames']):
                mask[j, ..., self._lines(h, r, t), :] = 1
        else:
            mask[..., self._lines(h, r, item.get('frame', 0)), :] = 1
        item['mask'] = mask
        item['ksub'] = mask * kfull
        return item


class DimExpansion(object):
    def __call__(self, item):
        item['ksub'] = torch.unsqueeze(item['ksub'], dim=0)
        item['imfull'] = torch.unsqueeze(item['imfull'], dim=0)
        item['mask'] = torch.unsqueeze(item['mask'], dim=0)
        if 'sens' in item:
            item['sens'] = torch.unsqueeze(item['sens'], dim=0)
        return item
