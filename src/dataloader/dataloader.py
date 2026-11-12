import os
import random
import h5py as h5
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import Compose
from .transforms import *

# Acceleration factors: training draws one at random per sample; validation
# reports each separately (the challenge uses R = 4, 8, 10).
TRAIN_R = (4, 6, 8, 10, 12)
VAL_R = (4, 8, 10)


class CMRDataset(Dataset):
    """One sample per frame. neighbors=n (2.5D) also loads the n frames
    before and after it (wrapping around: cine is periodic), giving kfull and
    sens of shape (2n + 1, ...) with the sample's own frame in the centre."""
    def __init__(self, dset_path, multi_coil, transform, sample_ratio=1.0,
                 neighbors=0):
        self.multi_coil = multi_coil
        self.transform = transform
        self.neighbors = neighbors
        self.map = []
        filenames = [filename for filename in os.listdir(dset_path)
                     if filename.endswith('.h5')]
        if sample_ratio > 1.0:
            raise ValueError('sample_ratio should be smaller than 1.0.')
        random.shuffle(filenames)
        filenames = filenames[:int(sample_ratio * len(filenames))]
        for filename in filenames:
            filename = os.path.join(dset_path, filename)
            with h5.File(filename, 'r') as f:
                num_slices = f['kspace'].shape[0]
            self.map += [(filename, i) for i in range(num_slices)]

    def __len__(self):
        return len(self.map)

    def __getitem__(self, idx):
        item = {}
        filename, i = self.map[idx]
        with h5.File(filename, 'r') as f:
            if self.neighbors:
                t = f['kspace'].shape[0]
                frames = [(i + d) % t for d in range(-self.neighbors,
                                                     self.neighbors + 1)]
                item['frames'] = frames
                item['kfull'] = np.stack([f['kspace'][j] for j in frames])
                if self.multi_coil:
                    item['sens'] = np.stack([f['sensitivity_map'][j]
                                             for j in frames])
            else:
                item['frame'] = i
                item['kfull'] = f['kspace'][i]
                if self.multi_coil:
                    item['sens'] = f['sensitivity_map'][i]
        return self.transform(item)


def _head(crop_readout):
    return [ToTensor(), CropReadout()] if crop_readout else [ToTensor()]


def _target(neighbors):
    return [TargetRecon()] + ([CenterFrame(neighbors)] if neighbors else [])


def get_dataloader(dset_path, is_training, multi_coil, sample_ratio=1.0,
                   batch_size=1, num_workers=4, crop_readout=True,
                   train_R=TRAIN_R, neighbors=0, pattern='fixed',
                   kt_sigma=0.25):
    if is_training:
        transform = [*_head(crop_readout), RandomFlip(), *_target(neighbors),
                     Subsampling(24, tuple(train_R), pattern, kt_sigma), DimExpansion()]
        return DataLoader(CMRDataset(
            dset_path, multi_coil, Compose(transform), sample_ratio,
            neighbors), batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True)
    else:
        datasets = []
        for subsample_ratios in ((4,), (8,), (10,)):
            transform = [*_head(crop_readout), *_target(neighbors),
                         Subsampling(24, subsample_ratios, pattern, kt_sigma, seed=0),
                         DimExpansion()]
            datasets.append(CMRDataset(
                dset_path, multi_coil, Compose(transform), sample_ratio,
                neighbors))
        return DataLoader(ConcatDataset(datasets), batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)


class CMRTemporalDataset(Dataset):
    """Each sample is a slice with its temporal frames.

    Stored kspace per file: (T, kH, kW) single-coil or (T, C, kH, kW)
    multi-coil (plus sensitivity_map of the same shape). After DimExpansion
    the DataLoader batches ksub to (B, 1, T, kH, kW) or (B, 1, T, C, kH, kW).
    clip_len=L > 0 splits each slice into clips of L consecutive frames
    (wrapping around; cine is periodic) instead of using all T frames.
    clip_len=1 trains a temporal model on single frames: same architecture,
    no temporal information (a control for the temporal prior).
    """
    def __init__(self, dset_path, transform, sample_ratio=1.0,
                 multi_coil=False, clip_len=0, read_sens=True):
        self.transform = transform
        self.read_sens = read_sens        # False: vista maps from the data
        self.multi_coil = multi_coil
        self.clip_len = clip_len
        filenames = [f for f in os.listdir(dset_path) if f.endswith('.h5')]
        if sample_ratio > 1.0:
            raise ValueError('sample_ratio should be smaller than 1.0.')
        random.shuffle(filenames)
        filenames = filenames[:int(sample_ratio * len(filenames))]
        self.map = []
        for f in filenames:
            f = os.path.join(dset_path, f)
            if clip_len:
                with h5.File(f, 'r') as h:
                    t = h['kspace'].shape[0]
                self.map += [(f, [(start + j) % t for j in range(clip_len)])
                             for start in range(0, t, clip_len)]
            else:
                self.map.append((f, None))

    def __len__(self):
        return len(self.map)

    def __getitem__(self, idx):
        item = {}
        filename, frames = self.map[idx]
        with h5.File(filename, 'r') as f:
            read = (lambda d: f[d][()]) if frames is None else \
                (lambda d: np.stack([f[d][j] for j in frames]))
            item['fname'] = filename
            item['nt'] = f['kspace'].shape[0]
            item['frames'] = list(range(item['nt'])) \
                if frames is None else frames
            item['kfull'] = read('kspace')
            item['multi_coil'] = self.multi_coil
            if self.multi_coil and self.read_sens:
                item['sens'] = read('sensitivity_map')
        return self.transform(item)


def get_temporal_dataloader(dset_path, is_training, sample_ratio=1.0,
                            batch_size=1, num_workers=4, crop_readout=True,
                            train_R=TRAIN_R, multi_coil=False, clip_len=0,
                            pattern='fixed', kt_sigma=0.25):
    """Dataloader for single-coil temporal (2+1)D models."""
    if is_training:
        transform = Compose([*_head(crop_readout), RandomFlip(), TargetRecon(),
                             Subsampling(24, tuple(train_R), pattern, kt_sigma), DimExpansion()])
        return DataLoader(CMRTemporalDataset(dset_path, transform, sample_ratio,
                               multi_coil, clip_len, not pattern.startswith('vista')),
                          batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    else:
        datasets = []
        for subsample_ratios in ((4,), (8,), (10,)):
            transform = Compose([*_head(crop_readout), TargetRecon(),
                                 Subsampling(24, subsample_ratios, pattern, kt_sigma,
                                             seed=0), DimExpansion()])
            datasets.append(CMRTemporalDataset(dset_path, transform, sample_ratio,
                               multi_coil, clip_len, not pattern.startswith('vista')))
        return DataLoader(ConcatDataset(datasets), batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)


def get_val_dataloaders(dset_path, multi_coil, sample_ratio=1.0, batch_size=1,
                        num_workers=4, crop_readout=True,
                        val_R=VAL_R, neighbors=0, pattern='fixed',
                        kt_sigma=0.25):
    """Returns {R: DataLoader} for per-acceleration-factor validation."""
    result = {}
    for r in val_R:
        transform = Compose([*_head(crop_readout), *_target(neighbors),
                             Subsampling(24, (r,), pattern, kt_sigma, seed=0),
                             DimExpansion()])
        result[r] = DataLoader(
            CMRDataset(dset_path, multi_coil, transform, sample_ratio,
                       neighbors),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True)
    return result


def get_temporal_val_dataloaders(dset_path, sample_ratio=1.0, batch_size=1,
                                 num_workers=4, crop_readout=True,
                                 val_R=VAL_R, multi_coil=False, clip_len=0,
                                 pattern='fixed', kt_sigma=0.25):
    """Returns {R: DataLoader} for per-R validation of temporal models."""
    result = {}
    for r in val_R:
        transform = Compose([*_head(crop_readout), TargetRecon(),
                             Subsampling(24, (r,), pattern, kt_sigma, seed=0),
                             DimExpansion()])
        result[r] = DataLoader(
            CMRTemporalDataset(dset_path, transform, sample_ratio,
                               multi_coil, clip_len, not pattern.startswith('vista')),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True)
    return result
