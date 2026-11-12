"""LV volumes (EDV, ESV) and ejection fraction on the CMRxRecon test subjects
(P112-P119, full short-axis stacks) from nnU-Net v1 Task027_ACDC (2d)
segmentations of the reference and of each reconstruction.

Geometry: the CMRxRecon files store no spacing; the published cine
parameters are used (Wang et al., Sci Data 2024: 1.5 x 1.5 mm in-plane,
8 mm slices; challenge page: 4 mm gap), i.e. voxel 1.5 x 1.5 mm and slice
spacing 12 mm. Volume = sum over slices of LV-cavity area x 12 mm (Simpson's
method of discs). EF does not depend on these values; EDV/ESV in mL scale
with them (equally for all methods).

Phases from the reference only: the reference is segmented in every frame;
ED = frame with the largest total LV volume, ES = the smallest. Each
reconstruction is then segmented at those two frames.

Steps (nnU-Net run in between with scripts/nnunet_predict_safe.py):
    python scripts/nnunet_cmrx_volumes.py export_ref --recon_dir REC --out IN_REF
    (predict IN_REF -> SEG_REF)
    python scripts/nnunet_cmrx_volumes.py export_rec --recon_dir REC \
        --ref_seg SEG_REF --out IN_REC
    (predict IN_REC -> SEG_REC)
    python scripts/nnunet_cmrx_volumes.py volumes --ref_seg SEG_REF \
        --rec_in IN_REC --rec_seg SEG_REC --out figures/cmrx_test_nnunet_volumes.csv
    (add --missed zero -> figures/cmrx_test_nnunet_volumes_zero.csv for the
    sensitivity analysis)

Missed cavities: a slice with an LV cavity in the reference but none in a
reconstruction is given the reference area by default (--missed ref), which
approximates manual correction and is optimistic; --missed zero keeps the
fully automated result (zero area).
"""
import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

PIXEL_MM = (1.5, 1.5)
SLICE_THICK_MM, SLICE_SPACING_MM = 8.0, 12.0
METHODS = ('stC3', 'DLE', 'CVN')
RS = (8, 10, 12, 16)


def write(img, path):
    im = sitk.GetImageFromArray(img[None].astype(np.float32))
    im.SetSpacing((PIXEL_MM[1], PIXEL_MM[0], SLICE_THICK_MM))
    sitk.WriteImage(im, path)


def norm(x):
    x = x.astype(np.float32)
    return x / x.max(axis=(1, 2), keepdims=True)


def slices(recon_dir):
    names = sorted({re.sub(r'_R\d+\.npz$', '', os.path.basename(p))
                    for p in glob.glob(os.path.join(recon_dir, '*_sax_*_R8.npz'))})
    return names


def lv_area_mm2(path):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(path))[0]
    m = seg == 3
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == int(np.argmax(sizes)) + 1
    return m.sum() * PIXEL_MM[0] * PIXEL_MM[1]


def export_ref(a):
    os.makedirs(a.out, exist_ok=True)
    for s in slices(a.recon_dir):
        ref = norm(np.load(os.path.join(a.recon_dir, f'{s}_R8.npz'))['target'])
        for t in range(ref.shape[0]):
            write(ref[t], os.path.join(a.out, f'{s}__target__f{t:02d}_0000.nii.gz'))
    print(len(os.listdir(a.out)), 'reference frames written')


def ref_volume_curves(ref_seg):
    rows = []
    for p in glob.glob(os.path.join(ref_seg, '*__target__f*.nii.gz')):
        s, _, f = os.path.basename(p)[:-7].split('__')
        rows.append(dict(subject=s.split('_')[0], slice=s, frame=int(f[1:]),
                         area=lv_area_mm2(p)))
    d = pd.DataFrame(rows)
    vol = d.groupby(['subject', 'frame']).area.sum() * SLICE_SPACING_MM / 1000  # mL
    return d, vol


def export_rec(a):
    _, vol = ref_volume_curves(a.ref_seg)
    phases = {}
    for subj, v in vol.groupby(level=0):
        v = v.droplevel(0)
        phases[subj] = dict(ED=int(v.idxmax()), ES=int(v.idxmin()))
    os.makedirs(a.out, exist_ok=True)
    json.dump(phases, open(os.path.join(a.out, 'phases.json'), 'w'), indent=1)
    for s in slices(a.recon_dir):
        ph = phases[s.split('_')[0]]
        for R in RS:
            z = np.load(os.path.join(a.recon_dir, f'{s}_R{R}.npz'))
            for m in METHODS:
                x = norm(z[m])
                for name, t in ph.items():
                    write(x[t], os.path.join(a.out, f'{s}__{m}__R{R}__{name}_0000.nii.gz'))
    print(phases)
    print(len(os.listdir(a.out)) - 1, 'reconstruction frames written')


def volumes(a):
    d, vol = ref_volume_curves(a.ref_seg)
    phases = json.load(open(os.path.join(a.rec_in, 'phases.json')))
    rows = []
    for subj, ph in phases.items():
        rows.append(dict(subject=subj, method='target', R=0,
                         EDV=vol[subj, ph['ED']], ESV=vol[subj, ph['ES']]))
    ref_area = {(row.slice, row.frame): row.area for row in d.itertuples()}
    rec = []
    for p in glob.glob(os.path.join(a.rec_seg, '*__R*__E?.nii.gz')):
        s, m, r, name = os.path.basename(p)[:-7].split('__')
        subj = s.split('_')[0]
        area, ra = lv_area_mm2(p), ref_area[s, phases[subj][name]]
        missed = ra > 0 and area == 0
        if missed and a.missed == 'ref':
            area = ra
        rec.append(dict(subject=subj, method=m, R=int(r[1:]), phase=name,
                        area=area, missed=missed, ref_cavity=ra > 0))
    r = pd.DataFrame(rec)
    n_ref = int(r[(r.method == r.method.iloc[0]) & (r.R == r.R.iloc[0])].ref_cavity.sum())
    print('missed cavities (reference cavity, none in the reconstruction; '
          f'handled as {a.missed!r}) of {n_ref} ED/ES slice-phases with a '
          'reference cavity:')
    print(r.groupby(['method', 'R']).missed.sum().unstack().to_string())
    v = (r.groupby(['subject', 'method', 'R', 'phase']).area.sum()
         * SLICE_SPACING_MM / 1000).unstack('phase').reset_index()
    v = v.rename(columns={'ED': 'EDV', 'ES': 'ESV'})
    out = pd.concat([pd.DataFrame(rows), v], ignore_index=True)
    out['EF'] = 100 * (out.EDV - out.ESV) / out.EDV
    out.to_csv(a.out, index=False)
    print(out[out.method == 'target'].round(1).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['export_ref', 'export_rec', 'volumes'])
    p.add_argument('--recon_dir')
    p.add_argument('--ref_seg')
    p.add_argument('--rec_in')
    p.add_argument('--rec_seg')
    p.add_argument('--missed', default='ref', choices=['ref', 'zero'],
                   help='volumes: a slice with an LV cavity in the reference '
                        'but none in the reconstruction gets the reference '
                        'area (ref: approximates manual correction, primary '
                        'analysis) or zero area (zero: fully automated '
                        'result, sensitivity analysis).')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    {'export_ref': export_ref, 'export_rec': export_rec, 'volumes': volumes}[a.step](a)


if __name__ == '__main__':
    main()
