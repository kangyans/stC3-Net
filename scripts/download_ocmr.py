"""Download fully sampled cine data of the OCMR dataset (external test set).

OCMR (Chen et al., arXiv:2008.03410, https://ocmr.info) is a public, open-
access multi-coil k-space cine dataset from The Ohio State University
(Siemens 0.55 T Free.Max, 1.5 T Avanto/Sola, 3 T Prisma/Vida). The files
are ISMRMRD .h5 in the public S3 bucket `ocmr`; no account is needed.

Files are selected from ocmr_data_attributes.csv:
    smp == fs (fully sampled), viw in --views, and by default without
    asymmetric echo (ech == asy) or an aliased field of view (fov == ali),
    which have no clean fully sampled reference.

    python download_ocmr.py --out /scratch/$USER/CMRxRecon/ocmr --dry-run
    python download_ocmr.py --out /scratch/$USER/CMRxRecon/ocmr --views sax

Re-running resumes: complete files (size equal to the S3 listing) are
skipped, partial ones are continued with an HTTP Range request.
"""
import os
import re
import csv
import time
import argparse
import urllib.parse
import urllib.request

BUCKET = 'https://ocmr.s3.amazonaws.com'
ATTRIBUTES = 'ocmr_data_attributes.csv'


def get_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', required=True,
                   help='Output directory (files go to <out>/raw).')
    p.add_argument('--views', nargs='+', default=['sax', 'lax'],
                   help='OCMR view labels (column viw, case-insensitive).')
    p.add_argument('--include_asym', default=False, action='store_true',
                   help='Also asymmetric-echo files (partial readout).')
    p.add_argument('--include_aliased', default=False, action='store_true',
                   help='Also files with an aliased field of view.')
    p.add_argument('--dry-run', dest='dry_run', default=False,
                   action='store_true', help='Only list the selection.')
    p.add_argument('--retries', type=int, default=5)
    return p


def list_bucket():
    """{key: size} of every object in the bucket."""
    sizes, token = {}, None
    while True:
        url = f'{BUCKET}/?list-type=2&max-keys=1000'
        if token:
            url += '&continuation-token=' + urllib.parse.quote(token, safe='')
        with urllib.request.urlopen(url, timeout=60) as f:
            xml = f.read().decode()
        for key, size in re.findall(
                r'<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>', xml):
            sizes[key] = int(size)
        m = re.search(r'<NextContinuationToken>([^<]+)<', xml)
        if not m:
            return sizes
        token = m.group(1)


def select(rows, views, include_asym, include_aliased):
    views = {v.lower() for v in views}
    out = []
    for r in rows:
        name = r['file name']
        if r['smp'] != 'fs' or r['viw'].lower() not in views:
            continue
        if r['ech'] == 'asy' and not include_asym:
            continue
        if r['fov'] == 'ali' and not include_aliased:
            continue
        out.append(name)
    return sorted(out)


def fetch(url, target, size, retries):
    for attempt in range(1, retries + 1):
        try:
            have = os.path.getsize(target) if os.path.exists(target) else 0
            if have == size:
                return 'skip'
            if have > size:
                os.remove(target)
                have = 0
            req = urllib.request.Request(url)
            if have:
                req.add_header('Range', f'bytes={have}-')
            with urllib.request.urlopen(req, timeout=120) as f, \
                    open(target, 'ab' if have else 'wb') as out:
                while True:
                    block = f.read(1 << 24)
                    if not block:
                        break
                    out.write(block)
            if os.path.getsize(target) == size:
                return 'ok'
            raise IOError(f'size {os.path.getsize(target)} != {size}')
        except Exception as ex:
            print(f'[x] {os.path.basename(target)} attempt {attempt}/'
                  f'{retries}: {ex}', flush=True)
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f'failed to download {url}')


def main():
    a = get_parser().parse_args()
    raw = os.path.join(a.out, 'raw')
    os.makedirs(raw, exist_ok=True)
    sizes = list_bucket()
    attr = os.path.join(a.out, ATTRIBUTES)
    fetch(f'{BUCKET}/{ATTRIBUTES}', attr, sizes[ATTRIBUTES], a.retries)
    with open(attr, newline='') as f:
        rows = list(csv.DictReader(f))
    names = select(rows, a.views, a.include_asym, a.include_aliased)
    missing = [n for n in names if f'data/{n}' not in sizes]
    names = [n for n in names if f'data/{n}' in sizes]
    total = sum(sizes[f'data/{n}'] for n in names)
    print(f'[o] {len(names)} files, {total / 1e9:.1f} GB (views {a.views}, '
          f'asym {a.include_asym}, aliased {a.include_aliased})', flush=True)
    if missing:
        print(f'[!] in the attributes table but not in the bucket: {missing}')
    if a.dry_run:
        for n in names:
            print(f'    {n} {sizes["data/" + n] / 1e6:.0f} MB')
        return
    for i, n in enumerate(names):
        status = fetch(f'{BUCKET}/data/{n}', os.path.join(raw, n),
                       sizes[f'data/{n}'], a.retries)
        print(f'[o] {i + 1}/{len(names)} {n}: {status}', flush=True)
    print('[o] Done.')


if __name__ == '__main__':
    main()
