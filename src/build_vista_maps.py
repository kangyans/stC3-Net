"""Precompute the VISTA coil maps that `--mask vista` training reads, so the
GPU jobs do not run ESPIRiT (~4 s per call) in the data loader.

For every slice file of --data_dir, every R and every flip, the slice goes
through the same steps as in the training loader (ToTensor, CropReadout,
flip, Subsampling 'vista'), which writes the maps to
<data_dir>/../vista_maps/. Maps that already exist are skipped.
Training flips randomly (4 states); validation does not flip (use --flips 0).

    python build_vista_maps.py --data_dir .../train --R 4 6 8 10
    python build_vista_maps.py --data_dir .../val --R 4 8 10 --flips 0
    # OCMR (frames have consistent phase): no frame phase alignment
    python build_vista_maps.py --data_dir .../ocmr/.../test --flips 0 --no_align
"""
import os
import argparse
import h5py
from dataloader.transforms import (ToTensor, CropReadout, Subsampling,
                                   flip_item, vista_map_path)

FLIPS = {'0': [], 'x': [-1], 'y': [-2], 'xy': [-1, -2]}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data_dir', required=True)
    p.add_argument('--R', type=int, nargs='+', default=[4, 6, 8, 10])
    p.add_argument('--flips', nargs='+', default=list(FLIPS),
                   choices=list(FLIPS))
    p.add_argument('--no_crop_readout', dest='crop_readout', default=True,
                   action='store_false')
    p.add_argument('--pattern', default='vista', choices=['vista', 'vista_dc'])
    p.add_argument('--no_align', dest='align', default=True,
                   action='store_false',
                   help='Time-average without frame phase alignment '
                        '(maps saved as *_noalign.npy).')
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--shard_id', type=int, default=0)
    a = p.parse_args()
    files = sorted(os.path.join(a.data_dir, f) for f in os.listdir(a.data_dir)
                   if f.endswith('.h5'))[a.shard_id::a.num_shards]
    print(f'[o] {len(files)} files, R {a.R}, flips {a.flips}', flush=True)
    for n, fname in enumerate(files):
        with h5py.File(fname, 'r') as f:
            kspace = f['kspace'][()]
        base = ToTensor()({'kfull': kspace})
        if a.crop_readout:
            base = CropReadout()(base)
        h, w = base['kfull'].shape[-2:]
        made = 0
        for flip in a.flips:
            dims = FLIPS[flip]
            todo = [r for r in a.R if not os.path.exists(vista_map_path(
                fname, r, h, w, flip.replace('0', ''), a.align, a.pattern))]
            if not todo:
                continue
            item = flip_item(dict(base), dims)
            for r in todo:
                Subsampling(24, (r,), a.pattern, vista_align=a.align)(dict(
                    item, fname=fname, nt=kspace.shape[0],
                    frames=list(range(kspace.shape[0])), multi_coil=True))
                made += 1
        print(f'[o] {n + 1}/{len(files)} {os.path.basename(fname)}: '
              f'{made} maps written', flush=True)


if __name__ == '__main__':
    main()
