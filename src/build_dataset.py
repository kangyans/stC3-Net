import os
import time
import argparse
from utils.io import *
from utils.recon import *


def get_parser():
    parser = argparse.ArgumentParser(
        description='Building training/validation dataset from raw data.')
    parser.add_argument('--old_root', type=str,
                        help='Root path of the raw data.')
    parser.add_argument('--new_root', type=str,
                        help='Root path of the new dataset.')
    parser.add_argument('--subject_range', type=int, nargs=2, default=None,
                        metavar=('START', 'END'),
                        help='Only use subjects [START, END) of the sorted '
                             'subject list, e.g. to hold out a val split.')
    parser.add_argument('--time_avg_maps', dest='per_frame_maps',
                        default=True, action='store_false',
                        help='Estimate one set of ESPIRiT maps from the '
                             'time-averaged k-space (old behaviour) instead '
                             'of one per frame.')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Split subjects into this many shards.')
    parser.add_argument('--shard_id', type=int, default=0,
                        help='Index of the shard processed by this run.')
    return parser


def build_dataset(old_root, new_root, multi_coil, num_shards=1, shard_id=0,
                  subject_range=None, per_frame_maps=True):
    if not os.path.exists(new_root):
        os.makedirs(new_root)
    subjects = sorted(subject for subject in os.listdir(old_root)
                      if os.path.isdir(os.path.join(old_root, subject)))
    if subject_range is not None:
        subjects = subjects[subject_range[0]:subject_range[1]]
    subjects = subjects[shard_id::num_shards]
    for i, subject in enumerate(subjects):
        print('Start processing subject {} ...'.format(subject))
        start_time = time.perf_counter()
        for data_file in ['cine_lax', 'cine_sax']:
            old_filename = os.path.join(
                old_root, subject, data_file + '.mat')
            if os.path.exists(old_filename):
                kfull = h5read(old_filename, 'kspace_full' if multi_coil
                               else 'kspace_single_full')
                kfull = kfull['real'] + 1j * kfull['imag']
                for s in range(kfull.shape[1]):
                    new_filename = os.path.join(
                        new_root, '{}_{}_slice{:02d}.h5'.format(
                            subject, data_file, s))
                    if os.path.exists(new_filename):
                        continue
                    tmp_filename = new_filename + '.tmp'
                    if os.path.exists(tmp_filename):
                        os.remove(tmp_filename)
                    kfull[:, s] /= np.max(np.abs(kfull[:, s]))
                    h5write(tmp_filename, 'kspace',
                            kfull[:, s].astype(np.csingle))
                    if multi_coil:
                        sens = espirit_maps(kfull[:, s], per_frame=per_frame_maps)
                        h5write(tmp_filename, 'sensitivity_map',
                                sens.astype(np.csingle))
                    os.replace(tmp_filename, new_filename)
        print('Finish processing subject {}: {:.3f} seconds.'.format(
            subject, time.perf_counter() - start_time))


def main():
    args = get_parser().parse_args()
    old_root = args.old_root
    new_root = args.new_root
    multi_coil = 'MultiCoil' in old_root
    build_dataset(old_root, new_root, multi_coil,
                  args.num_shards, args.shard_id, args.subject_range,
                  args.per_frame_maps)


if __name__ == '__main__':
    main()
