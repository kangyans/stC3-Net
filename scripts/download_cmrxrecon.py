"""Download CMRxRecon 2023 data from Synapse.

Three releases are available (--dataset):
    challenge        syn53183714  Original 2023 challenge data, the layout this
                                  repo's code expects (MultiCoil|SingleCoil/
                                  Cine|Mapping/TrainingSet/FullSample, ...):
                                  ChallengeData.tar.gz-part-00 ... part-60,
                                  README.md, Format_*.txt, SubmissionFormat_*.txt
    challenge_after  syn53183715  Challenge validation/test sets released after
                                  the competition: ChallengeData_validation/_test
                                  .tar.gz-part-*, README.md
    cmrxrecon300     syn52965326  Raw scanner data (GRAPPA R=3 k-space + separate
                                  24-line calibration, no fully sampled
                                  reference): TrainingSet/ValidationSet/TestSet
                                  .tar.gz-part-*, DemoData, T1map_TI, tree_*.txt

Prerequisites
    1. Create a Synapse account and accept the CMRxRecon data use terms at
       https://www.synapse.org/#!Synapse:syn51471091/wiki/
    2. Create a personal access token (Account Settings -> Personal Access
       Tokens, with "Download" scope) and export it:
           export SYNAPSE_AUTH_TOKEN=<token>
    3. pip install synapseclient

Examples
    # list what would be downloaded (no login needed)
    python download_cmrxrecon.py --out /scratch/$USER/CMRxRecon --dry-run
    # only the docs of the challenge release
    python download_cmrxrecon.py --out /scratch/$USER/CMRxRecon \
        --subsets README Format SubmissionFormat
    # the whole challenge release, extracted
    python download_cmrxrecon.py --out /scratch/$USER/CMRxRecon --extract

Re-running the script is safe: parts that are already on disk with the
correct MD5 are skipped, so an interrupted download simply resumes.
"""
import os
import re
import json
import time
import hashlib
import fnmatch
import tarfile
import argparse
import subprocess
import urllib.request
from collections import defaultdict

REPO = 'https://repo-prod.prod.sagebase.org/repo/v1'
DATASETS = {'challenge': 'syn53183714',
            'challenge_after': 'syn53183715',
            'cmrxrecon300': 'syn52965326'}
# Default --subsets per dataset (file-name prefixes); None means everything.
DEFAULT_SUBSETS = {'challenge': None,
                   'challenge_after': None,
                   'cmrxrecon300': ['TrainingSet', 'ValidationSet', 'tree']}


def get_parser():
    parser = argparse.ArgumentParser(
        description='Download the CMRxRecon 2023 dataset from Synapse.')
    parser.add_argument('--out', type=str, required=True,
                        help='Output root directory.')
    parser.add_argument('--dataset', type=str, default='challenge',
                        choices=sorted(DATASETS),
                        help='Which Synapse release to download.')
    parser.add_argument('--subsets', type=str, nargs='+', default=None,
                        help='Only files whose names start with one of these '
                             'prefixes (default: depends on --dataset).')
    parser.add_argument('--extract', default=False, action='store_true',
                        help='Concatenate parts and extract after download.')
    parser.add_argument('--members', type=str, nargs='+', default=None,
                        help='With --extract: only extract archive members '
                             'whose path matches one of these shell patterns '
                             '(fnmatch, * also matches /), e.g. '
                             "'*/MultiCoil/Cine/*'. Default: everything.")
    parser.add_argument('--delete_parts', default=False, action='store_true',
                        help='Delete tarball parts after a successful extract.')
    parser.add_argument('--dry-run', dest='dry_run', default=False,
                        action='store_true',
                        help='Only list the files, do not download.')
    parser.add_argument('--retries', type=int, default=5,
                        help='Retries per file on network errors.')
    return parser


def rest_get(path):
    with urllib.request.urlopen(REPO + path) as f:
        return json.load(f)


def list_items(dataset_id):
    """Return [(entity_id, version, name)] of the files in the dataset."""
    dataset = rest_get(f'/entity/{dataset_id}')
    items = []
    for it in dataset['items']:
        e = rest_get(f"/entity/{it['entityId']}/version/{it['versionNumber']}")
        items.append((it['entityId'], it['versionNumber'], e['name']))
    return sorted(items, key=lambda x: x[2])


def selected(name, subsets):
    return subsets is None or any(name.startswith(s) for s in subsets)


def md5sum(filename, chunk=1 << 24):
    h = hashlib.md5()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


class NoAccessError(Exception):
    pass


def download(syn, entity_id, version, name, parts_dir, retries):
    target = os.path.join(parts_dir, name)
    for attempt in range(1, retries + 1):
        try:
            entity = syn.get(entity_id, version=version, downloadFile=False)
            expected = entity._file_handle.get('contentMd5')
            if os.path.exists(target) and expected and \
                    md5sum(target) == expected:
                print(f'[=] {name} already downloaded, skip.')
                return
            syn.get(entity_id, version=version, downloadLocation=parts_dir,
                    ifcollision='overwrite.local')
            if not os.path.exists(target):
                # synapseclient only warns (no exception) when the account
                # lacks DOWNLOAD permission, and returns no file.
                raise NoAccessError(
                    f'{name} was not downloaded: no DOWNLOAD permission. '
                    f'Request access at https://www.synapse.org/Synapse:'
                    f'{entity_id} (or its parent dataset) and accept the terms.')
            if expected and md5sum(target) != expected:
                raise IOError(f'MD5 mismatch for {name}')
            print(f'[o] {name} downloaded.')
            return
        except NoAccessError:
            raise
        except Exception as ex:
            print(f'[x] {name} attempt {attempt}/{retries} failed: {ex}')
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f'Failed to download {name}')


def extract_members(paths, out_dir, members):
    """Stream the joined parts through tarfile and extract only the members
    matching `members`. Returns (matched files, total files)."""
    cat = subprocess.Popen(['cat'] + paths, stdout=subprocess.PIPE)
    n_match = n_total = 0
    with tarfile.open(fileobj=cat.stdout, mode='r|gz') as tf:
        for m in tf:
            if m.isfile():
                n_total += 1
            if not any(fnmatch.fnmatch(m.name, pat) for pat in members):
                continue
            if m.isfile():
                n_match += 1
                if n_match % 500 == 0:
                    print(f'    {n_match} files extracted ({m.name})',
                          flush=True)
            if hasattr(tarfile, 'data_filter'):
                tf.extract(m, out_dir, filter='data')
            else:
                tf.extract(m, out_dir)
    cat.stdout.close()
    if cat.wait() != 0:
        raise RuntimeError('cat failed')
    return n_match, n_total


def extract(parts_dir, out_dir, delete_parts, members=None, archives=None):
    groups = defaultdict(list)
    for name in sorted(os.listdir(parts_dir)):
        m = re.match(r'(.+\.tar\.gz)-part-(\d+)$', name)
        if m:
            groups[m.group(1)].append(name)
        elif name.endswith('.tar.gz'):
            groups[name].append(name)
        elif name.endswith('.zip'):
            groups[name].append(name)
    tag = 'all' if members is None else ' '.join(members)
    for archive, parts in groups.items():
        if archives is not None and archive not in archives:
            continue
        # Parts must be joined in numeric order (part-2 before part-10).
        parts.sort(key=lambda n: int(re.search(r'(\d+)$', n).group(1))
                   if re.search(r'-part-\d+$', n) else 0)
        done_flag = os.path.join(out_dir, f'.{archive}.extracted')
        if os.path.exists(done_flag) and \
                open(done_flag).read().strip() in ('', tag, 'all'):
            print(f'[=] {archive} already extracted ({tag}), skip.')
            continue
        print(f'[o] Extracting {archive} ({len(parts)} parts, '
              f'members: {tag}) ...', flush=True)
        paths = [os.path.join(parts_dir, p) for p in parts]
        if archive.endswith('.zip'):
            cmd = ['unzip', '-q', '-o', paths[0], '-d', out_dir]
            subprocess.run(cmd, check=True)
        elif members is not None:
            n_match, n_total = extract_members(paths, out_dir, members)
            print(f'[o] {archive}: {n_match} of {n_total} files matched.')
            if n_match == 0:
                raise RuntimeError(
                    f'no member of {archive} matches {members}; check the '
                    f'layout with: cat {paths[0]} | tar -tzf - | head')
        else:
            # Stream the parts through tar without writing the joined file.
            cat = subprocess.Popen(['cat'] + paths, stdout=subprocess.PIPE)
            subprocess.run(['tar', '-xzf', '-', '-C', out_dir],
                           stdin=cat.stdout, check=True)
            cat.stdout.close()
            if cat.wait() != 0:
                raise RuntimeError(f'cat failed for {archive}')
        with open(done_flag, 'w') as f:
            f.write(tag)
        if delete_parts:
            for p in paths:
                os.remove(p)


def main():
    args = get_parser().parse_args()
    parts_dir = os.path.join(args.out, 'parts')
    os.makedirs(parts_dir, exist_ok=True)

    dataset_id = DATASETS[args.dataset]
    subsets = args.subsets or DEFAULT_SUBSETS[args.dataset]
    items = [it for it in list_items(dataset_id) if selected(it[2], subsets)]
    print(f'[o] {len(items)} files selected from {args.dataset} ({dataset_id}):')
    for entity_id, version, name in items:
        print(f'    {entity_id}.{version}  {name}')
    if args.dry_run:
        return

    import warnings
    import synapseclient
    # syn.get() is deprecated in synapseclient 4.11+ (removed in 5.0, which
    # requirements.txt excludes); silence the per-file warning spam.
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    syn = synapseclient.Synapse(skip_checks=True)
    syn.login(authToken=os.environ.get('SYNAPSE_AUTH_TOKEN'), silent=True)
    for entity_id, version, name in items:
        try:
            download(syn, entity_id, version, name, parts_dir, args.retries)
        except NoAccessError as ex:
            raise SystemExit(f'[x] {ex}')

    if args.extract:
        archives = {re.sub(r'-part-\d+$', '', name) for _, _, name in items}
        extract(parts_dir, args.out, args.delete_parts, args.members,
                archives)
    print('[o] Done.')


if __name__ == '__main__':
    main()
