"""Save side-by-side PNGs of a trained model's reconstructions.

    python scripts/preview_recon.py <exp_dir> [--val_dir DIR] [--R 4 8 10]
                                    [--num 6] [--ckpt best_ckpt.pt] [--use_gpu]

<exp_dir> is a train.py output folder (with args.json and the checkpoint).
For each R, --num validation images (spread evenly over the set) are written
to <exp_dir>/preview/R<R>_<i>.png, all scaled by the target's max:
    top row:    GT | zero-filled (IFFT) | ESPIRiT | network,
                each titled with its challenge-style SSIM / PSNR
    bottom row: 5 x |error| of each reconstruction
The box on GT marks the central region the challenge scores (and the metrics
use). ESPIRiT appears for multi-coil models (temporal: middle frame).
ESPIRiT is the classical baseline: CG-SENSE with the dataset's ESPIRiT maps,
12 iterations, Tikhonov 0.01 (as in DemoCode/ReconDemo). Challenge-style
SSIM/PSNR/NMSE per image are printed, averaged per R, and saved to
<exp_dir>/preview/metrics.csv.
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import skimage.io as io
from PIL import Image, ImageDraw, ImageFont
from sigpy.mri.app import SenseRecon
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))
import train                                  # noqa: E402
from dataloader.dataloader import (get_val_dataloaders,  # noqa: E402
                                   get_temporal_val_dataloaders)
from networks.losses import SSIMLoss          # noqa: E402


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('exp_dir', type=str)
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Validation data (default: from args.json).')
    parser.add_argument('--R', type=int, nargs='+', default=None,
                        help='Acceleration factors (default: --val_R used '
                             'in training).')
    parser.add_argument('--num', type=int, default=6,
                        help='Images per R.')
    parser.add_argument('--ckpt', type=str, default='best_ckpt.pt')
    parser.add_argument('--use_gpu', default=False, action='store_true')
    parser.add_argument('--no_espirit', dest='espirit', default=True,
                        action='store_false',
                        help='Skip the ESPIRiT (CG-SENSE) baseline.')
    parser.add_argument('--espirit_lamda', type=float, default=0.01,
                        help='Tikhonov weight for the ESPIRiT baseline '
                             '(0.01 as in the challenge demo; larger values '
                             'help at high R and hurt at R=4).')
    return parser


def build_net(a):
    multi_coil = 'MultiCoil' in a['trn_dir']
    net = train.get_net(
        a['real'], multi_coil, a['temporal'], a['cascade_depth'], a['lamda'],
        a['net_depth'], a['num_filters'], a['kernel_size'], a['bias'],
        a['normalization'], a['activation'], a['down'], a['up'],
        num_coils=a['num_coils'] if a.get('coil_mode') == 'channel' else None,
        final_combine=a.get('final_combine', 'sense'),
        neighbors=a.get('neighbors', 0), stc3v2=train.stc3v2_options(a))
    train.apply_model_options(net, a)
    return net, multi_coil


def middle_frame(x, ndim):
    """Take the middle frame along dim 2 until x has ndim dims
    (temporal models: all tensors; 2.5D: the inputs)."""
    while x.dim() > ndim:
        x = x[:, :, x.shape[2] // 2]
    return x


def zero_filled(ksub, multi_coil):
    im = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(
        ksub, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
    if multi_coil:
        return torch.linalg.vector_norm(im, dim=-3)         # RSS over coils
    return im.abs()


def espirit_recon(ksub, sens, lamda=0.01, max_iter=12):
    """CG-SENSE with ESPIRiT maps. ksub, sens: (1, 1, C, H, W) -> (1, 1, H, W).

    Sampled locations are taken from the non-zero k-space entries; sigpy's
    centred orthonormal FFT matches the data's convention.
    """
    y = ksub[0, 0].cpu().numpy()
    mps = sens[0, 0].cpu().numpy()
    x = SenseRecon(y, mps, lamda=lamda, max_iter=max_iter,
                   show_pbar=False).run()
    return torch.from_numpy(np.abs(x)).float()[None, None].to(ksub.device)


def to_panel(x, scale):
    return np.clip(x.squeeze().cpu().numpy() / scale, 0, 1)


def _font(size):
    try:
        return ImageFont.load_default(size=size)     # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def challenge_box(h, w, readout_factor):
    """(top, left, height, width) of the region _challenge_metrics scores."""
    mround = lambda x: int(np.floor(x + 0.5))
    ch, cw = mround(h / 2), min(w, mround(w * readout_factor / 3))
    return h // 2 - ch // 2, w // 2 - cw // 2, ch, cw


def make_figure(columns, title, box, zoom=2):
    """columns: [(name, image, error_or_None, metrics_or_None)], images in
    [0, 1]. Returns an RGB uint8 array: titled image row over an error row."""
    h, w = columns[0][1].shape
    H, W, head, gap = h * zoom, w * zoom, 22 * zoom, 4
    top = 18 * zoom
    fig = Image.new('RGB', (len(columns) * (W + gap) - gap,
                            top + 2 * (head + H) + gap), 'black')
    draw = ImageDraw.Draw(fig)
    big, small = _font(9 * zoom), _font(8 * zoom)
    draw.text((4, 2), title, fill='white', font=big)
    for c, (name, im, err, met) in enumerate(columns):
        x0 = c * (W + gap)
        label = name if met is None else \
            f'{name}   SSIM {met[0]:.4f}   PSNR {met[1]:.2f} dB'
        draw.text((x0 + 4, top + 2), label,
                  fill='white' if met is None else (255, 220, 120), font=small)
        tile = Image.fromarray((255 * im).astype(np.uint8)).resize(
            (W, H), Image.BICUBIC).convert('RGB')
        fig.paste(tile, (x0, top + head))
        if c == 0 and box is not None:
            t, l, bh, bw = (v * zoom for v in box)
            draw.rectangle([x0 + l, top + head + t, x0 + l + bw,
                            top + head + t + bh], outline=(80, 200, 255))
        if err is not None:
            y0 = top + head + H + gap
            draw.text((x0 + 4, y0 + 2), f'{name}: 5x |error|',
                      fill=(170, 170, 170), font=small)
            tile = Image.fromarray((255 * err).astype(np.uint8)).resize(
                (W, H), Image.BICUBIC).convert('RGB')
            fig.paste(tile, (x0, y0 + head))
    return np.asarray(fig)


def main():
    args = get_parser().parse_args()
    with open(os.path.join(args.exp_dir, 'args.json')) as f:
        a = json.load(f)
    val_dir = args.val_dir or a['val_dir']
    rs = args.R or a.get('val_R', [4, 8, 10])
    device = 'cuda' if args.use_gpu else 'cpu'

    net, multi_coil = build_net(a)
    ckpt = torch.load(os.path.join(args.exp_dir, args.ckpt),
                      map_location=device)
    net.load_state_dict(ckpt['net'])
    net.to(device).eval()
    print(f'[o] {args.exp_dir}: epoch {ckpt["epoch"] + 1}, '
          f'best val loss {ckpt["best_val_loss"]:.4f}')

    crop = a.get('crop_readout', False)
    random.seed(0)        # datasets shuffle their file lists: same picks every run
    pattern = a.get('mask') or ('interleave' if a.get('interleave') else 'fixed')
    sampling = dict(pattern=pattern, kt_sigma=a.get('kt_sigma', 0.25))
    if a['temporal']:
        loaders = get_temporal_val_dataloaders(val_dir, 1.0, 1, 0, crop, rs,
                                               multi_coil, a.get('clip_len', 0),
                                               **sampling)
    else:
        loaders = get_val_dataloaders(val_dir, multi_coil, 1.0, 1, 0, crop,
                                      rs, neighbors=a.get('neighbors', 0),
                                      **sampling)
    out_dir = os.path.join(args.exp_dir, 'preview')
    os.makedirs(out_dir, exist_ok=True)
    ssim_fn = SSIMLoss().to(device)
    factor = 2 if crop else 1
    use_esp = args.espirit and multi_coil
    kdim = 5 if multi_coil else 4       # (B, 1, [C,] kH, kW) once frames removed
    methods = ['net', 'zf'] + (['esp'] if use_esp else [])
    rows = ['R,index,' + ','.join(f'{m}_{q}' for m in methods
                                   for q in ('ssim', 'psnr', 'nmse'))]
    summary = {}

    for r, dl in loaders.items():
        ds = dl.dataset
        picks = np.linspace(0, len(ds) - 1, min(args.num, len(ds))).astype(int)
        for i, item in zip(picks, DataLoader(Subset(ds, picks.tolist()),
                                             batch_size=1)):
            item = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in item.items()}
            with torch.no_grad():
                if 'sens' in item:
                    out = net(item['ksub'], item['mask'], item['sens'])
                else:
                    out = net(item['ksub'], item['mask'])
            tgt = middle_frame(item['imfull'], 4)
            ksub = middle_frame(item['ksub'], kdim)
            ims = {'net': middle_frame(out, 4),
                   'zf': zero_filled(ksub, multi_coil)}
            if use_esp:
                ims['esp'] = espirit_recon(ksub, middle_frame(item['sens'], 5),
                                           args.espirit_lamda)
            m = {k: train._challenge_metrics(ssim_fn, v, tgt, factor)[0]
                 for k, v in ims.items()}
            for k in methods:
                summary.setdefault((r, k), []).append(m[k])

            scale = tgt.max().item()
            names = {'zf': 'Zero-filled (IFFT)', 'esp': 'ESPIRiT',
                     'net': 'Network'}
            order = ['zf'] + (['esp'] if use_esp else []) + ['net']
            columns = [('GT', to_panel(tgt, scale), None, None)] + [
                (names[k], to_panel(ims[k], scale),
                 to_panel(5 * (ims[k] - tgt).abs(), scale), m[k])
                for k in order]
            box = challenge_box(*tgt.shape[-2:], factor)
            panel = make_figure(
                columns, f'R = {r}   |   validation image #{i}   |   '
                f'{os.path.basename(os.path.normpath(args.exp_dir))}', box)
            png = os.path.join(out_dir, f'R{r:02d}_{i:05d}.png')
            io.imsave(png, panel, check_contrast=False)
            rows.append(f'{r},{i},' + ','.join(
                f'{m[k][0]:.4f},{m[k][1]:.2f},{m[k][2]:.4f}' for k in methods))
            print(f'    R={r:2d} #{i:5d}: SSIM net {m["net"][0]:.4f}'
                  + (f' | ESPIRiT {m["esp"][0]:.4f}' if use_esp else '')
                  + f' | zero-filled {m["zf"][0]:.4f}'
                  + f'  -> {os.path.basename(png)}')
    with open(os.path.join(out_dir, 'metrics.csv'), 'w') as f:
        f.write('\n'.join(rows) + '\n')

    names = {'zf': 'zero-filled', 'esp': 'ESPIRiT', 'net': 'network'}
    print(f'[o] Mean over {args.num} images per R (SSIM / PSNR dB / NMSE):')
    for r in loaders:
        print(f'    R={r:2d}: ' + ' | '.join(
            f'{names[k]} {np.mean([x[0] for x in summary[(r, k)]]):.4f} / '
            f'{np.mean([x[1] for x in summary[(r, k)]]):.2f} / '
            f'{np.mean([x[2] for x in summary[(r, k)]]):.4f}'
            for k in ['zf', 'esp', 'net'] if (r, k) in summary))
    print(f'[o] Wrote {len(rows) - 1} previews to {out_dir}')


if __name__ == '__main__':
    main()
