"""Convert fully sampled OCMR cine (ISMRMRD .h5) into the per-slice .h5 files
that the CMRxRecon data loader reads, for use as an external test set.

Output per slice, the same format as build_dataset.py writes for CMRxRecon:
    kspace           (T, 10, ky, kx) complex64, centred, readout (last axis)
                     with its 2x oversampling kept (the loader crops it),
                     scaled to max |k| = 1
    sensitivity_map  (T, 10, ky, kx) complex64, per-frame ESPIRiT (24 lines)
    attrs            OCMR file, scanner, field strength, view, coils, ...
File names: <out_dir>/ocmr_<id>_<field>_<view>_slice<ss>.h5

Steps per slice:
  1. Read the imaging acquisitions (noise, navigator, phase-correction,
     dummy and feedback scans are skipped; reversed lines are flipped) and
     place them on the encoded matrix: ky = step1 - centre + Ny/2 with
     Ny = 2 * max(centre - min, max - centre + 1) over the acquired range,
     kx = Nx/2 - centre_sample (fills an asymmetric echo from its centre).
     Repeated averages of the same (frame, ky) are averaged.
  2. SVD coil compression to --num_coils virtual coils (the models take
     exactly 10 coils as channels; CMRxRecon was also compressed to 10).
  3. Scale, per-frame ESPIRiT, write.
The frame count stays native (OCMR has 15-40+ frames, CMRxRecon 12).

    python build_ocmr.py --raw_dir .../ocmr/raw --out_dir \\
        .../built/ocmr/MultiCoil/Cine/test --attributes .../ocmr_data_attributes.csv
"""
import os
import re
import csv
import json
import time
import argparse
import h5py
import numpy as np
from utils.recon import espirit_maps

# ISMRMRD flags are 1-based bit positions: value = 1 << (flag - 1).
SKIP_FLAGS = [19, 23, 24, 26, 27, 28, 29]  # noise, navigation, phase corr.,
# HP feedback, dummy scan, RT feedback, surface coil correction
SKIP_MASK = sum(1 << (f - 1) for f in SKIP_FLAGS)
REVERSE = 1 << (22 - 1)


def get_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--raw_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--attributes', default=None,
                   help='ocmr_data_attributes.csv; adds scanner/view/... to '
                        'the attrs and applies the --views filter.')
    p.add_argument('--views', nargs='+', default=['sax', 'lax'])
    p.add_argument('--num_coils', type=int, default=10)
    p.add_argument('--keep_partial', default=False, action='store_true',
                   help='Also write slices with unacquired (frame, ky) lines '
                        '(partial Fourier); skipped by default because the '
                        'reference is then not fully sampled.')
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--shard_id', type=int, default=0)
    return p


def xml_float(xml, path):
    """Number at a nested tag path such as encodedSpace/matrixSize/x."""
    block = xml
    for tag in path.split('/'):
        m = re.search(rf'<{tag}>(.*?)</{tag}>', block, re.S)
        if m is None:
            raise KeyError(path)
        block = m.group(1)
    return float(block.strip())


def xml_int(xml, path):
    return int(round(xml_float(xml, path)))


def read_ocmr(filename):
    """Return k-space (slices, frames, coils, ky, kx), the ky lines that were
    acquired, and header info."""
    with h5py.File(filename, 'r') as f:
        xml = f['dataset/xml'][0]
        xml = xml.decode() if isinstance(xml, bytes) else xml
        dset = f['dataset/data']
        head = dset['head']
        nx = xml_int(xml, 'encodedSpace/matrixSize/x')
        rx = xml_int(xml, 'reconSpace/matrixSize/x')
        os_x = xml_float(xml, 'encodedSpace/fieldOfView_mm/x') / \
            xml_float(xml, 'reconSpace/fieldOfView_mm/x')
        if abs(os_x - 2) > 0.01:
            raise ValueError(f'readout oversampling {os_x:.3f}, loader '
                             f'crops by exactly 2')
        # Phase encodes: the acquired range, symmetric about its centre.
        # (encodedSpace y can include the scanner's interpolation, e.g. 294
        # for 104 acquired lines.) A partial-Fourier side stays zero.
        lo = xml_int(xml, 'encodingLimits/kspace_encoding_step_1/minimum')
        hi = xml_int(xml, 'encodingLimits/kspace_encoding_step_1/maximum')
        cy = xml_int(xml, 'encodingLimits/kspace_encoding_step_1/center')
        ny = 2 * max(cy - lo, hi - cy + 1)
        idx = head['idx']
        keep = (head['flags'].astype(np.uint64) & np.uint64(SKIP_MASK)) == 0
        for k in ['contrast', 'repetition', 'set']:
            vals = np.unique(idx[k][keep])
            if len(vals) > 1:
                raise ValueError(f'{k} has {len(vals)} values; not supported')
        ns = int(idx['slice'][keep].max()) + 1
        nt = int(idx['phase'][keep].max()) + 1
        nc = int(head['active_channels'][keep].max())
        k = np.zeros((ns, nt, nc, ny, nx), np.complex64)
        cnt = np.zeros((ns, nt, ny), np.int32)
        rows = np.where(keep)[0]
        for start in range(0, len(rows), 1024):
            chunk = rows[start:start + 1024]
            data = dset[chunk[0]:chunk[-1] + 1]
            for r in chunk:
                h = head[r]
                d = data[r - chunk[0]]['data'].view(np.complex64)
                n = int(h['number_of_samples'])
                d = d.reshape(nc, n)
                if int(h['flags']) & REVERSE:
                    d = d[:, ::-1]
                s, t = int(h['idx']['slice']), int(h['idx']['phase'])
                y = int(h['idx']['kspace_encode_step_1']) - cy + ny // 2
                x0 = nx // 2 - int(h['center_sample'])
                if not (0 <= y < ny and 0 <= x0 and x0 + n <= nx):
                    raise ValueError(f'line out of the encoded matrix: '
                                     f'y={y}, x={x0}..{x0 + n}')
                k[s, t, :, y, x0:x0 + n] += d
                cnt[s, t, y] += 1
    k /= np.maximum(cnt, 1)[:, :, None, :, None]
    info = dict(nx=nx, ny=ny, recon_x=rx, readout_os=round(os_x, 4),
                pe_limits=[lo, hi, cy],
                header_ny=xml_int(xml, 'encodedSpace/matrixSize/y'),
                coils=nc,
                field_T=float(re.search(r'<systemFieldStrength_T>(.*?)<',
                                        xml).group(1)),
                samples=int(head['number_of_samples'][keep].max()))
    return k, cnt > 0, info


def coil_compress(k, num_coils):
    """SVD coil compression of (T, C, ky, kx) to num_coils virtual coils."""
    t, c = k.shape[:2]
    if c <= num_coils:
        raise ValueError(f'{c} coils <= {num_coils}')
    a = np.moveaxis(k, 1, 0).reshape(c, -1)
    u, s, _ = np.linalg.svd(a @ a.conj().T)
    kc = (u[:, :num_coils].conj().T @ a).reshape(num_coils, t, *k.shape[2:])
    kept = float(np.sum(s[:num_coils]) / np.sum(s))
    return np.moveaxis(kc, 0, 1).astype(np.complex64), kept


def main():
    a = get_parser().parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    attrs = {}
    if a.attributes:
        with open(a.attributes, newline='') as f:
            attrs = {r['file name']: r for r in csv.DictReader(f)}
    views = {v.lower() for v in a.views}
    files = []
    for name in sorted(os.listdir(a.raw_dir)):
        if not name.endswith('.h5'):
            continue
        r = attrs.get(name)
        if r is not None and r['viw'].lower() not in views:
            continue
        files.append(name)
    files = files[a.shard_id::a.num_shards]
    print(f'[o] {len(files)} files in shard {a.shard_id}/{a.num_shards}',
          flush=True)
    for i, name in enumerate(files):
        t0 = time.perf_counter()
        r = attrs.get(name, {})
        stem = 'ocmr_' + name[len('fs_'):-len('.h5')] if name.startswith(
            'fs_') else os.path.splitext(name)[0]
        view = (r.get('viw') or 'unk').lower().replace(' ', '')
        k, acq, info = read_ocmr(os.path.join(a.raw_dir, name))
        ns, nt = k.shape[:2]
        for s in range(ns):
            out = os.path.join(a.out_dir, f'{stem}_{view}_slice{s:02d}.h5')
            if os.path.exists(out):
                continue
            missing = int((~acq[s]).sum())
            if missing:
                print(f'[!] {name} slice {s}: {missing} of {acq[s].size} '
                      f'(frame, ky) lines not acquired' +
                      ('' if a.keep_partial else ', skipped'), flush=True)
                if not a.keep_partial:
                    continue
            kc, energy = coil_compress(k[s], a.num_coils)
            kc /= np.max(np.abs(kc))
            sens = espirit_maps(kc, per_frame=True).astype(np.complex64)
            tmp = out + '.tmp'
            with h5py.File(tmp, 'w') as f:
                f.create_dataset('kspace', data=kc)
                f.create_dataset('sensitivity_map', data=sens)
                meta = dict(info, ocmr_file=name, slice=s, frames=nt,
                            coil_energy_kept=round(energy, 5),
                            missing_lines=missing,
                            **{key: r.get(key, '') for key in
                               ['scn', 'ech', 'viw', 'sli', 'fov', 'sub']})
                f.attrs['ocmr'] = json.dumps(meta)
            os.replace(tmp, out)
        print(f'[o] {i + 1}/{len(files)} {name}: {ns} slice(s), {nt} frames, '
              f'{info["coils"]}->{a.num_coils} coils, matrix '
              f'{info["ny"]}x{info["nx"]}, {time.perf_counter() - t0:.0f} s',
              flush=True)


if __name__ == '__main__':
    main()
