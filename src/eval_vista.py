"""Test-set evaluation of trained models under VISTA sampling (true R).

Every slice of --data_dir (optionally filtered by --subjects) is undersampled
with the same deterministic VISTA pattern and coil maps (time-averaged
ESPIRiT from the undersampled data) as in validation, and reconstructed by
each model and by two non-learned baselines:
    zf       zero-filled, RSS over coils
    espirit  CG-SENSE with the same VISTA coil maps (sigpy SenseRecon,
             lamda 0.01, 12 iterations; as scripts/preview_recon.py)
    ttv      l1-ESPIRiT with temporal total variation (utils/ttv.py, ADMM)
             with the same maps; frame phases removed with the same
             estimate as the maps (none with --no_align)
Metrics are the challenge-style per-image SSIM / PSNR / NMSE used in
validation (train._challenge_metrics: central crop, each image scaled by its
own maximum), one row per (slice, frame, R, method) in --out (CSV).

Use the final-epoch checkpoint (ckpt.pt, default): best_ckpt.pt was chosen
by the validation loss on P111-P120 and must not be used for a test set
that overlaps them.

    # CMRxRecon test subjects (P111 was the development subject)
    python eval_vista.py --data_dir $BUILT/challenge/MultiCoil/Cine/val \\
        --subjects 'P11[2-9]|P120' --runs stC3=$EXP/vista_stC3 \\
        DLE=$EXP/vista_DLE CVN=$EXP/vista_CVN --out results_cmrx.csv --use_gpu
    # OCMR external test (frames without phase flips: --no_align)
    python eval_vista.py --data_dir $BUILT/ocmr/MultiCoil/Cine/test \\
        --dataset ocmr --no_align --runs ... --out results_ocmr.csv --use_gpu
    # summary + paired Wilcoxon tests over subjects from an existing CSV
    python eval_vista.py --summarize results_cmrx.csv --reference stC3

Rerunning with the same --out skips (slice, R, method) already in the CSV,
so an interrupted job resumes.
"""
import os
import re
import csv
import sys
import json
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
from torchvision.transforms import Compose

import train
from networks.losses import SSIMLoss
from networks.stc3 import STC3Net
from utils.ttv import ttv_recon
from utils.vista import frame_phases
from dataloader.dataloader import CMRTemporalDataset
from dataloader.transforms import (ToTensor, CropReadout, TargetRecon,
                                   Subsampling, DimExpansion)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import preview_recon as P                                      # noqa: E402

FIELDS = ['dataset', 'subject', 'file', 'view', 'scanner', 'field_T', 'R',
          'R_true', 'method', 'frame', 'ssim', 'psnr', 'nmse', 'time_s']


def get_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data_dir', help='Directory of slice .h5 files.')
    p.add_argument('--runs', nargs='+', default=[],
                   help='label=experiment_dir (holds args.json and the '
                        'checkpoint). Append @stc3_output=mag to evaluate a '
                        'stC3 checkpoint with the output |x| of the last '
                        'cascade instead of the RSS of its data-consistent '
                        'k-space (inference only; same weights).')
    p.add_argument('--ckpt', default='ckpt.pt',
                   help='Checkpoint file name in each experiment dir.')
    p.add_argument('--R', type=int, nargs='+', default=[4, 6, 8, 10, 12, 16])
    p.add_argument('--baselines', nargs='*', default=['zf', 'espirit', 'ttv'],
                   choices=['zf', 'espirit', 'ttv'])
    # TTV defaults tuned on the development subject P111 (challenge SSIM):
    # lam 0.01 (R8 grid 0.001-0.1: 0.879-0.915, best 0.01-0.03), rho 1;
    # 50 ADMM x 5 CG iterations (100 ADMM changes SSIM by <= 0.007)
    p.add_argument('--ttv_lambda', type=float, default=0.01,
                   help='TTV weight (data scaled to max |E^H y| = 1).')
    p.add_argument('--ttv_rho', type=float, default=1.0)
    p.add_argument('--ttv_iters', type=int, default=50,
                   help='ADMM iterations (5 CG steps each).')
    p.add_argument('--subjects', default=None,
                   help='Regex on the file name, e.g. "P11[2-9]|P120".')
    p.add_argument('--dataset', default='cmrx', help='Tag for the CSV.')
    p.add_argument('--no_align', dest='align', default=True,
                   action='store_false',
                   help='No frame phase alignment in the coil-map and stC3 '
                        'SPIRiT calibration (OCMR).')
    p.add_argument('--pattern', default=None, choices=['vista', 'vista_dc'],
                   help='Sampling pattern (default: the --mask of the runs, '
                        'which must agree; vista without runs).')
    p.add_argument('--out', help='Output CSV (appended to, resumable).')
    p.add_argument('--save_recon', default=None,
                   help='Directory: also save magnitude reconstructions '
                        '(float16 .npz per slice and R) for figures.')
    p.add_argument('--use_gpu', default=False, action='store_true')
    p.add_argument('--summarize', default=None, metavar='CSV',
                   help='Only print the summary of an existing CSV.')
    p.add_argument('--reference', default=None,
                   help='--summarize: method compared with every other one '
                        '(paired Wilcoxon over subjects).')
    return p


def load_run(path, ckpt, device):
    with open(os.path.join(path, 'args.json')) as f:
        a = json.load(f)
    net, multi_coil = P.build_net(a)
    if not multi_coil or not a.get('temporal') or \
            not str(a.get('mask')).startswith('vista'):
        print(f'[!] {path}: multi_coil={multi_coil}, temporal='
              f'{a.get("temporal")}, mask={a.get("mask")} (expected '
              f'multi-coil temporal VISTA)', flush=True)
    state = torch.load(os.path.join(path, ckpt), map_location=device)
    net.load_state_dict(state['net'])
    print(f'[o] {path}/{ckpt}: epoch {state.get("epoch", -1) + 1}, '
          f'{sum(p.numel() for p in net.parameters())} parameters', flush=True)
    return net.to(device).eval(), a


def slice_info(fname, dataset):
    """subject, view, scanner, field strength of a slice file."""
    base = os.path.basename(fname)[:-3]
    info = dict(subject=base.split('_')[0], view='', scanner='', field_T='')
    m = re.search(r'cine_(sax|lax)', base)
    if m:
        info['view'] = m.group(1)
    try:
        import h5py
        with h5py.File(fname, 'r') as f:
            if 'ocmr' in f.attrs:
                meta = json.loads(f.attrs['ocmr'])
                info.update(subject='ocmr_' + meta['ocmr_file'][3:7],
                            view=meta.get('viw', '').lower(),
                            scanner=meta.get('scn', ''),
                            field_T=round(meta.get('field_T', 0), 2))
    except OSError:
        pass
    if dataset == 'cmrx' and not info['scanner']:
        info.update(scanner='cmrxrecon', field_T=3.0)
    return info


def done_keys(path):
    keys = set()
    if path and os.path.exists(path):
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                keys.add((r['file'], int(r['R']), r['method']))
    return keys


def evaluate(a):
    device = 'cuda' if a.use_gpu else 'cpu'
    runs, masks = [], set()
    for spec in a.runs:
        label, path = spec.split('=', 1)
        path, _, override = path.partition('@')
        net, args = load_run(path, a.ckpt, device)
        if isinstance(net, STC3Net):
            net.calib_align = a.align
        if override:
            key, val = override.split('=', 1)
            if key != 'stc3_output' or val not in ('rss', 'mag') or \
                    not isinstance(net, STC3Net):
                raise SystemExit(f'unsupported override {override!r} for {label}')
            print(f'[o] {label}: output {net.output} -> {val} (inference only)',
                  flush=True)
            net.output = val
        runs.append((label, net))
        masks.add(args.get('mask'))
    if a.pattern is None:
        if len(masks) > 1:
            raise SystemExit(f'runs trained with different masks {masks}; '
                             f'give --pattern')
        a.pattern = masks.pop() if masks else 'vista'
    ssim_fn = SSIMLoss().to(device)
    done = done_keys(a.out)
    new = not (a.out and os.path.exists(a.out))
    out = open(a.out, 'a', newline='') if a.out else None
    writer = csv.DictWriter(out, FIELDS) if out else None
    if writer and new:
        writer.writeheader()
    methods = list(a.baselines) + [label for label, _ in runs]
    print(f'[o] methods {methods}, R {a.R}, pattern {a.pattern}, '
          f'align {a.align}', flush=True)

    for r in a.R:
        transform = Compose([ToTensor(), CropReadout(), TargetRecon(),
                             Subsampling(24, (r,), a.pattern, seed=0,
                                         vista_align=a.align),
                             DimExpansion()])
        ds = CMRTemporalDataset(a.data_dir, transform, multi_coil=True,
                                read_sens=False)
        ds.map = sorted(m for m in ds.map if a.subjects is None or
                        re.search(a.subjects, os.path.basename(m[0])))
        print(f'[o] R={r}: {len(ds)} slices', flush=True)
        sums = defaultdict(list)
        for i in range(len(ds)):
            fname = ds.map[i][0]
            todo = [m for m in methods
                    if (os.path.basename(fname), r, m) not in done]
            if not todo:
                continue
            item = ds[i]
            to = lambda x: x[None].to(device)
            ksub, mask = to(item['ksub']), to(item['mask'])
            sens, tgt = to(item['sens']), to(item['imfull'])
            h = ksub.shape[-2]
            r_true = h / float((mask[0, 0, 0, 0, :, 0] > 0).sum())
            info = slice_info(fname, a.dataset)
            recons = {}
            for m in todo:
                t0 = time.perf_counter()
                with torch.no_grad():
                    if m == 'zf':
                        im = P.zero_filled(ksub, True)
                    elif m == 'ttv':
                        k0, m0, s0 = ksub[0, 0], mask[0, 0], sens[0, 0]
                        ph = frame_phases(k0) if a.align else None
                        im = ttv_recon(k0, m0, s0, a.ttv_lambda, a.ttv_rho,
                                       a.ttv_iters, phases=ph).abs()[None, None]
                    elif m == 'espirit':
                        im = torch.cat([P.espirit_recon(ksub[:, :, t],
                                                        sens[:, :, t])
                                        for t in range(ksub.shape[2])], 0)
                        im = im.permute(1, 0, 2, 3)[None]   # (1, 1, T, H, W)
                    else:
                        im = dict(runs)[m](ksub, mask, sens)
                if a.use_gpu:
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                im = im.reshape(tgt.shape)
                metrics = train._challenge_metrics(ssim_fn, im, tgt, 2)
                for t, (s, p, n) in enumerate(metrics):
                    if writer:
                        writer.writerow(dict(
                            dataset=a.dataset, file=os.path.basename(fname),
                            R=r, R_true=round(r_true, 3), method=m, frame=t,
                            ssim=round(s, 5), psnr=round(p, 3),
                            nmse=round(n, 6), time_s=round(dt, 3), **info))
                sums[m].append(np.mean([x[0] for x in metrics]))
                recons[m] = im
            if out:
                out.flush()
            if a.save_recon:
                os.makedirs(a.save_recon, exist_ok=True)
                np.savez_compressed(
                    os.path.join(a.save_recon, f'{os.path.basename(fname)[:-3]}'
                                 f'_R{r}.npz'),
                    target=tgt.squeeze().cpu().numpy().astype(np.float16),
                    **{m: v.squeeze().cpu().numpy().astype(np.float16)
                       for m, v in recons.items()})
            print(f'    {i + 1}/{len(ds)} {os.path.basename(fname)} R_true '
                  f'{r_true:.2f}: ' + ' '.join(
                      f'{m} {sums[m][-1]:.3f}' for m in todo), flush=True)
    if out:
        out.close()


def summarize(path, reference=None):
    """Mean +- SD over subjects (each subject = mean over its images) per
    dataset, field strength, view, R and method; paired Wilcoxon signed-rank tests over subjects
    against `reference`."""
    rows = list(csv.DictReader(open(path, newline='')))
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r['dataset'], r.get('field_T', ''), r.get('view', ''),
               int(r['R']), r['method'])
        groups[key][r['subject']].append(
            (float(r['ssim']), float(r['psnr']), float(r['nmse'])))
    per_subj = {k: {s: np.mean(v, 0) for s, v in d.items()}
                for k, d in groups.items()}
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    print(f'{"dataset":8s} {"field":>5s} {"view":4s} {"R":>3s} {"method":12s} {"n":>3s} '
          f'{"SSIM":>15s} {"PSNR":>13s} {"NMSE":>15s}  vs {reference}')
    for key in sorted(per_subj):
        d = per_subj[key]
        v = np.array(list(d.values()))
        line = (f'{key[0]:8s} {key[1]:>5s} {key[2]:4s} {key[3]:3d} {key[4]:12s} '
                f'{len(v):3d} {v[:, 0].mean():.4f}+-{v[:, 0].std():.4f} '
                f'{v[:, 1].mean():6.2f}+-{v[:, 1].std():4.2f} '
                f'{v[:, 2].mean():.4f}+-{v[:, 2].std():.4f}')
        ref = per_subj.get(key[:4] + (reference,))
        if reference and ref and key[4] != reference and wilcoxon:
            subj = sorted(set(d) & set(ref))
            diff = [ref[s][0] - d[s][0] for s in subj]
            if len(subj) >= 5 and any(diff):
                p = wilcoxon([ref[s][0] for s in subj],
                             [d[s][0] for s in subj]).pvalue
                line += f'  dSSIM {np.mean(diff):+.4f} p={p:.2g} (n={len(subj)})'
        print(line)


def main():
    a = get_parser().parse_args()
    if a.summarize:
        summarize(a.summarize, a.reference)
        return
    if not a.data_dir:
        raise SystemExit('--data_dir is required')
    evaluate(a)
    if a.out:
        summarize(a.out, a.runs[0].split('=')[0] if a.runs else None)


if __name__ == '__main__':
    main()
