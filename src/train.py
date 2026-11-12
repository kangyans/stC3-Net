import sys
import os
import json
import math
import time
import shutil
import argparse

import torch.nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from networks.models import *
from networks.losses import *
from networks.stc3 import STC3Net
from dataloader.dataloader import (get_dataloader, get_temporal_dataloader,
                                   get_val_dataloaders, get_temporal_val_dataloaders)


def get_parser():
    parser = argparse.ArgumentParser(description='Training CMRxRecon model.')
    parser.add_argument('--trn_dir', type=str,
                        help='Directory of training data.')
    parser.add_argument('--val_dir', type=str,
                        help='Directory of validation data.')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Data sample ratio.')
    parser.add_argument('--real', default=False, action='store_true',
                        help='Real-valued or complex-valued network.')
    parser.add_argument('--cascade_depth', type=int, default=2,
                        help='Depth of cascade model, 0 or 1 is special, '
                             '0 means k-space domain model,'
                             '1 means image domain model.')
    parser.add_argument('--lamda', type=float, default=0.99,
                        help='Data consistency regularization parameter.')
    parser.add_argument('--net_depth', type=int, default=4,
                        help='Depth of network.')
    parser.add_argument('--num_filters', type=int, default=64,
                        help='Number of filters of intermediate layers.')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size of convolutional layers.')
    parser.add_argument('--bias', default=False, action='store_true',
                        help='Add bias or not for convolutional layers.')
    parser.add_argument('--normalization', type=str,
                        choices=['instance', 'batch'], default='instance',
                        help='Type of normalization, should be '
                             'instance or batch.')
    parser.add_argument('--activation', type=str,
                        choices=['relu', 'leakyrelu'], default='relu',
                        help='Type of activation, should be '
                             'relu or leakyrelu')
    parser.add_argument('--down', type=str,
                        choices=['avg', 'max'], default='avg',
                        help='Type of downsampling block, should be '
                             'avg or max.')
    parser.add_argument('--up', type=str,
                        choices=['bilinear', 'convtran'], default='bilinear',
                        help='Type of upsampling block, should be '
                             'bilinear or convtran.')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate.')
    parser.add_argument('--use_gpu', default=False, action='store_true',
                        help='Use GPU for training.')
    parser.add_argument('--temporal', default=False, action='store_true',
                        help='Use (2+1)D temporal model on whole slices (complex; '
                             'single- or multi-coil).')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training and validation.')
    parser.add_argument('--scheduler', type=str,
                        choices=['cosine', 'none'], default='cosine',
                        help='Learning rate scheduler.')
    parser.add_argument('--freq_weight', '--kspace_weight', dest='freq_weight',
                        type=float, default=0.0,
                        help='Weight of an L1 loss between the 2D FFT '
                             'magnitudes of the output and target magnitude '
                             'images (a frequency-domain image loss, not a '
                             'loss on the measured k-space). Off by default.')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join('..', 'experiments'),
                        help='Root directory for experiment outputs.')
    parser.add_argument('--coil_mode', type=str, default='channel',
                        choices=['channel', 'batch'],
                        help='Multi-coil k-space nets: all coils as channels '
                             'in one pass (channel) or each coil separately '
                             '(batch, the original behaviour).')
    parser.add_argument('--num_coils', type=int, default=10,
                        help='Coil count for --coil_mode channel.')
    parser.add_argument('--final_combine', type=str, default='rss',
                        choices=['rss', 'sense'],
                        help='Final coil combination of multi-coil k-space '
                             'and cascade models: rss (matches the target) '
                             'or sense (original behaviour).')
    parser.add_argument('--train_R', type=int, nargs='+',
                        default=[4, 6, 8, 10, 12],
                        help='Acceleration factors for training; each sample '
                             'uses one at random (always + 24 centre lines).')
    parser.add_argument('--val_R', type=int, nargs='+', default=[4, 8, 10],
                        help='Acceleration factors validated separately '
                             'each epoch.')
    parser.add_argument('--dc', type=str, default='blend',
                        choices=['blend', 'varnet', 'grad', 'cg'],
                        help='Data consistency: blend (shared lamda at '
                             'sampled points) or varnet (learned gradient '
                             'step per cascade, E2E-VarNet style); for '
                             '--stc3v2: grad (learned gradient step on the '
                             'ESPIRiT model) or cg (CG-SENSE after the image '
                             'net).')
    parser.add_argument('--condition_R', default=False, action='store_true',
                        help='Condition every UNet on the acceleration '
                             'factor read from the mask (FiLM).')
    parser.add_argument('--grad_checkpoint', default=False,
                        action='store_true',
                        help='Recompute UNet activations in backward to save '
                             'memory (deep cascades / temporal models).')
    parser.add_argument('--stc3', default=False, action='store_true',
                        help='stC^3-Net default configuration: --stc3v2 '
                             'with the ACS-calibrated k-net (--knet_domain '
                             'acs); sizes from STC3_SIZES unless given.')
    parser.add_argument('--norm_stats', type=str, default='running',
                        choices=['running', 'instance'],
                        help='Instance-norm statistics at evaluation: '
                             'running (track_running_stats, the original '
                             'behaviour: evaluation uses averages of recent '
                             'training samples) or instance (each sample\'s '
                             'own statistics, as in training; --stc3 '
                             'default).')
    parser.add_argument('--stc3v1', default=False, action='store_true',
                        help='stC^3-Net v1 (the former --stc3): (2+1)D '
                             'complex cross-domain cascade (--temporal) with '
                             'the shallow complex k-net (--knet shallow), '
                             'complex (2+1)D image UNet, blend data '
                             'consistency (--dc blend) and circular time '
                             'padding; sizes from STC3_SIZES unless given.')
    parser.add_argument('--stc3v2', default=False, action='store_true',
                        help='stC^3-Net v2 (networks/stc3.py): image-domain '
                             'state with ESPIRiT-model data consistency, '
                             'shallow k-net, (2+1)D image UNet, RSS of the '
                             'data-consistent k-space as output; sizes from '
                             'STC3_SIZES unless given.')
    parser.add_argument('--knet_domain', type=str, default='coil',
                        choices=['coil', 'combined', 'none', 'acs'],
                        help='--stc3v2 k-net input: per-coil k-space (coils '
                             'as channels), k-space of the coil-combined '
                             'image, no k-net, or per-coil k-space with an '
                             'ACS-calibrated SPIRiT kernel (acs).')
    parser.add_argument('--inet_real', default=False, action='store_true',
                        help='--stc3v2 ablation: real-valued (2+1)D image '
                             'UNets on [Re, Im] (width sqrt(2) x num_filters, '
                             'about the same parameters) instead of complex.')
    parser.add_argument('--knet_pos', default=False, action='store_true',
                        help='--stc3v2/--stc3: k-space position (k_y, k_x, '
                             'radius) as extra k-net input channels.')
    parser.add_argument('--knet_linear', default=False, action='store_true',
                        help='--stc3v2: k-net without normalisation and '
                             'activations (linear k-t kernel).')
    parser.add_argument('--phase_equivariant', default=False,
                        action='store_true',
                        help='--stc3v2: modReLU and magnitude instance norm '
                             'instead of CReLU / CInstanceNorm3d.')
    parser.add_argument('--phase_aug', default=False, action='store_true',
                        help='--stc3v2: random global input phase during '
                             'training.')
    parser.add_argument('--stc3_output', type=str, default='rss',
                        choices=['rss', 'mag'],
                        help='--stc3v2 output: RSS of the data-consistent '
                             'multi-coil k-space, or |x|.')
    parser.add_argument('--complex_weight', type=float, default=0.0,
                        help='Weight of an L1 loss on the complex image '
                             '(|x - S^H F^-1 kfull|; --stc3v2 only).')
    parser.add_argument('--knet', type=str, default='unet',
                        choices=['unet', 'shallow'],
                        help='k-net of the (2+1)D cross-domain cascade: the '
                             '(2+1)D UNet, or a shallow full-resolution '
                             'complex (2+1)D conv stack (no pooling).')
    parser.add_argument('--knet_width', type=int, default=32)
    parser.add_argument('--knet_layers', type=int, default=5)
    parser.add_argument('--knet_t_kernel', type=int, default=3,
                        help='Temporal taps of the shallow k-net '
                             '(1 = 2D per-frame k-space kernels).')
    parser.add_argument('--knet_mask', default=False, action='store_true',
                        help='Give the k-net the sampling mask as an extra '
                             'input channel.')
    parser.add_argument('--mask', type=str, default='fixed',
                        choices=['fixed', 'interleave', 'ktgaussian', 'vista',
                                 'vista_dc'],
                        help='Undersampling pattern: fixed (challenge: same '
                             'lines in every frame), interleave (frame t '
                             'samples lines from t mod R), ktgaussian '
                             '(per-frame variable-density random lines, same '
                             'line count as fixed), vista (VISTA, Ahmad et '
                             'al. 2015, at the TRUE R: no extra ACS lines; '
                             'coil maps calibrated from the phase-aligned '
                             'time average of the undersampled data; temporal '
                             'models only), vista_dc (VISTA with the k-space '
                             'centre line in every frame, same lines per '
                             'frame: the frame phase estimate then links all '
                             'frames). R of fixed/interleave/ktgaussian '
                             'is nominal (24 ACS lines on top).')
    parser.add_argument('--kt_sigma', type=float, default=0.25,
                        help='ktgaussian: std of the line density, as a '
                             'fraction of the phase-encode size.')
    parser.add_argument('--interleave', default=False, action='store_true',
                        help='Alias for --mask interleave.')
    parser.add_argument('--no_se', default=False, action='store_true',
                        help='Remove the squeeze-and-excitation blocks.')
    parser.add_argument('--image_only', default=False, action='store_true',
                        help='Cascade models without k-space nets: image-'
                             'domain cascades + data consistency.')
    parser.add_argument('--neighbors', type=int, default=0,
                        help='2.5D: give the 2D multi-coil cascade the n '
                             'frames before and after each frame as extra '
                             'input channels (0 = plain 2D).')
    parser.add_argument('--temporal_kernel', type=int, default=3,
                        help='Temporal kernel size of (2+1)D models. Note: '
                             'instance norm and SE blocks still pool over '
                             'all frames, so 1 is not time-blind (use '
                             '--clip_len 1 for that).')
    parser.add_argument('--clip_len', type=int, default=0,
                        help='Temporal models: train/validate on clips of '
                             'this many frames (0 = whole slice; 1 = single '
                             'frames, i.e. no temporal information).')
    parser.add_argument('--temporal_padding', type=str, default='zeros',
                        choices=['zeros', 'circular'],
                        help='Padding along time in (2+1)D models; circular '
                             'treats the cine sequence as periodic.')
    parser.add_argument('--no_residual', dest='residual', default=True,
                        action='store_false',
                        help='UNets output UNet(x) instead of x + UNet(x) '
                             '(original behaviour; starts below zero-filled).')
    parser.add_argument('--no_crop_readout', dest='crop_readout',
                        default=True, action='store_false',
                        help='Keep the 2x readout oversampling (default: '
                             'crop the central half of the readout).')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment folder name (default: built from '
                             'the architecture args).')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader worker processes.')
    return parser


# stC^3-Net default sizes: 8 cascades; complex (2+1)D image UNet with 16
# filters and 3 pooling levels; shallow complex (2+1)D k-net, 5 x 16 channels.
STC3_SIZES = {'cascade_depth': 8, 'num_filters': 16, 'net_depth': 3,
              'knet_width': 16, 'knet_layers': 5, 'knet_t_kernel': 3}


def stc3v2_options(a):
    """STC3Net options from train.py args (a dict), or None if not
    --stc3v2."""
    if not a.get('stc3v2', False):
        return None
    return dict(knet_width=a['knet_width'], knet_layers=a['knet_layers'],
                knet_t_kernel=a['knet_t_kernel'],
                knet_domain=a['knet_domain'], knet_linear=a['knet_linear'],
                knet_pos=a.get('knet_pos', False),
                dc=a['dc'], output=a['stc3_output'],
                phase_equivariant=a['phase_equivariant'],
                phase_aug=a['phase_aug'],
                inet_real=a.get('inet_real', False))


def get_net(real, multi_coil, temporal, cascade_depth, lamda, net_depth,
            num_filters, kernel_size, bias, normalization, activation, down, up,
            num_coils=None, final_combine='rss', neighbors=0, stc3v2=None):
    if stc3v2 is not None:      # dict of STC3Net options
        if not multi_coil or not temporal:
            raise ValueError('--stc3v2 needs multi-coil --temporal data.')
        return STC3Net(cascade_depth, net_depth, num_filters,
                       num_coils=num_coils or 10, **stc3v2)
    if neighbors:
        if not multi_coil or real or temporal or cascade_depth < 2:
            raise ValueError('--neighbors needs a complex multi-coil 2D '
                             'cascade (--cascade_depth >= 2).')
        return MultiCoilComplex25DCascadeNet(
            1, 1, cascade_depth, net_depth, num_filters, kernel_size, bias,
            normalization, activation, down, up, lamda,
            num_coils=num_coils or 10, num_frames=2 * neighbors + 1,
            final_combine=final_combine)
    if temporal and multi_coil:
        if real:
            raise ValueError('multi-coil temporal models are complex only.')
        if cascade_depth <= 1:
            return MultiCoilComplexTemporalImageDomainNet(
                1, 1, net_depth, num_filters, kernel_size,
                bias, normalization, activation, down, up)
        return MultiCoilComplexTemporalCascadeCrossDomainNet(
            1, 1, cascade_depth, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up, lamda,
            num_coils=num_coils or 10, final_combine=final_combine)
    if temporal:
        if cascade_depth <= 1:
            return SingleCoilComplexTemporalImageDomainNet(
                1, 1, net_depth, num_filters, kernel_size,
                bias, normalization, activation, down, up)
        else:
            return SingleCoilComplexTemporalCascadeCrossDomainNet(
                1, 1, cascade_depth, net_depth, num_filters,
                kernel_size, bias, normalization, activation,
                down, up, lamda)
    if real:
        if multi_coil:
            if cascade_depth == 0:
                return MultiCoilKspaceDomainNet(
                    2, 2, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up,
                    num_coils=num_coils,
                    final_combine=final_combine)
            elif cascade_depth == 1:
                return MultiCoilImageDomainNet(
                    2, 2, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            else:
                return MultiCoilCascadeCrossDomainNet(
                    2, 2, cascade_depth, net_depth, num_filters,
                    kernel_size, bias, normalization, activation,
                    down, up, lamda, num_coils=num_coils,
                    final_combine=final_combine)
        else:
            if cascade_depth == 0:
                return SingleCoilKspaceDomainNet(
                    2, 2, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            elif cascade_depth == 1:
                return SingleCoilImageDomainNet(
                    2, 2, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            else:
                return SingleCoilCascadeCrossDomainNet(
                    2, 2, cascade_depth, net_depth, num_filters,
                    kernel_size, bias, normalization, activation,
                    down, up, lamda)
    else:
        if multi_coil:
            if cascade_depth == 0:
                return MultiCoilComplexKspaceDomainNet(
                    1, 1, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up,
                    num_coils=num_coils,
                    final_combine=final_combine)
            elif cascade_depth == 1:
                return MultiCoilComplexImageDomainNet(
                    1, 1, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            else:
                return MultiCoilComplexCascadeCrossDomainNet(
                    1, 1, cascade_depth, net_depth, num_filters,
                    kernel_size, bias, normalization, activation,
                    down, up, lamda, num_coils=num_coils,
                    final_combine=final_combine)
        else:
            if cascade_depth == 0:
                return SingleCoilComplexKspaceDomainNet(
                    1, 1, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            elif cascade_depth == 1:
                return SingleCoilComplexImageDomainNet(
                    1, 1, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            else:
                return SingleCoilComplexCascadeCrossDomainNet(
                    1, 1, cascade_depth, net_depth, num_filters,
                    kernel_size, bias, normalization, activation,
                    down, up, lamda)


def apply_model_options(net, a):
    """Post-construction model options from train.py args (a dict, e.g. a
    loaded args.json). Missing keys mean the original behaviour."""
    set_knet(net, a.get('knet', 'unet'), a.get('knet_width', 32),
             a.get('knet_layers', 5), a.get('knet_t_kernel', 3),
             a.get('knet_mask', False))
    if a.get('image_only', False):
        make_image_only(net)
    if a.get('no_se', False):
        remove_se(net)
    if a.get('residual', False):
        make_residual(net)
    set_dc_mode(net, a.get('dc', 'blend'))
    if a.get('condition_R', False):
        make_conditional(net)
    if a.get('grad_checkpoint', False):
        set_grad_checkpoint(net)
        if isinstance(net, STC3Net):
            net.grad_checkpoint = True
        for m in net.modules():
            if isinstance(m, ShallowCKNet2p1d):
                m.grad_checkpoint = True
    set_temporal_conv(net, a.get('temporal_kernel', 3),
                      a.get('temporal_padding', 'zeros'))
    if a.get('norm_stats', 'running') == 'instance':
        use_instance_stats(net)
    return net


def use_instance_stats(net):
    """Instance norms use each sample's own statistics in training and
    evaluation alike (no running buffers, so none in the checkpoint)."""
    for m in net.modules():
        if isinstance(m, torch.nn.modules.instancenorm._InstanceNorm):
            m.track_running_stats = False
            m.running_mean = m.running_var = m.num_batches_tracked = None


def _magnitude(x):
    """Return real-valued magnitude, collapsing temporal dim into batch for 5-D."""
    x = x.abs() if x.is_complex() else x
    if x.dim() == 5:
        B, C, T, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
    return x


def _ssim_loss(ssim_fn, imcnn, imfull):
    """Compute SSIMLoss, collapsing the temporal dim into the batch dim if 5-D.

    The data range is each target image's own maximum (the challenge scores
    every image after normalising it by its maximum). A fixed range of 1.0
    is too large: targets peak around 0.5, which inflates SSIM's stability
    constants and makes the loss lenient in low-signal regions.
    """
    if imcnn.dim() == 5:
        B, C, T, H, W = imcnn.shape
        imcnn = imcnn.reshape(B * T, C, H, W)
        imfull = imfull.reshape(B * T, C, H, W)
    data_range = imfull.amax(dim=(-3, -2, -1)).clamp_min(1e-8)
    return ssim_fn(imcnn, imfull, data_range)


def _challenge_metrics(ssim_fn, imcnn, imfull, readout_factor):
    """Per-image (SSIM, PSNR, NMSE) as the CMRxRecon challenge scores them.

    Each 2D image is cropped to the central 1/2 of the phase-encode axis and
    1/3 of the original readout axis (readout_factor: how much the readout
    was already cropped, 2 with --crop_readout), then prediction and target
    are each divided by their own maximum. See
    DemoCode/EvaluationDemo/Evaluation.py and ChallengeDataFormat.
    """
    pred, tgt = _magnitude(imcnn), _magnitude(imfull)      # (N, 1, H, W)
    H, W = tgt.shape[-2:]
    mround = lambda x: int(math.floor(x + 0.5))          # MATLAB round()
    ch, cw = mround(H / 2), min(W, mround(W * readout_factor / 3))
    h0, w0 = H // 2 - ch // 2, W // 2 - cw // 2          # MATLAB crop() centre
    pred = pred[..., h0:h0 + ch, w0:w0 + cw]
    tgt = tgt[..., h0:h0 + ch, w0:w0 + cw]
    pred = pred / pred.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    tgt = tgt / tgt.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    one = tgt.new_tensor(1.0)
    metrics = []
    for p, t in zip(pred, tgt):
        mse = torch.mean((p - t) ** 2).item()
        metrics.append((1.0 - ssim_fn(p[None], t[None], one).item(),
                        -10.0 * math.log10(mse) if mse > 0 else 100.0,
                        (torch.norm(p - t) ** 2 / torch.norm(t) ** 2).item()))
    return metrics


def count_parameters(net):
    total_parameters = sum(p.numel() for p in net.parameters())
    trainable_parameters = sum(p.numel() for p in net.parameters()
                               if p.requires_grad)
    return total_parameters, trainable_parameters


def save_ckpt(ckpt_filename, net, optimizer, scheduler, best_val_loss, epoch):
    torch.save({'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'epoch': epoch}, f=ckpt_filename + '.tmp')
    # Atomic replace so a job killed mid-save never leaves a corrupt ckpt.
    os.replace(ckpt_filename + '.tmp', ckpt_filename)


def load_ckpt(ckpt_filename, net, optimizer, scheduler):
    ckpt = torch.load(ckpt_filename)
    net.load_state_dict(ckpt['net'])
    optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and ckpt.get('scheduler'):
        scheduler.load_state_dict(ckpt['scheduler'])
    return net, optimizer, scheduler, ckpt['best_val_loss'], ckpt['epoch'] + 1


def train(net, optimizer, dataloader, writer, epoch, use_gpu,
          l1_fn, ssim_fn, fft2c_fn, kspace_weight, complex_weight=0.0):
    start_time_epoch = start_time_step = time.perf_counter()

    net.train()
    avg_loss = 0.0
    global_step = epoch * len(dataloader)
    for step, item in enumerate(dataloader):
        ksub   = item['ksub'].to('cuda')   if use_gpu else item['ksub']
        mask   = item['mask'].to('cuda')   if use_gpu else item['mask']
        imfull = item['imfull'].to('cuda') if use_gpu else item['imfull']
        if 'sens' in item:
            sens = item['sens'].to('cuda') if use_gpu else item['sens']
            imcnn = net(ksub, mask, sens)
        else:
            imcnn = net(ksub, mask)

        l1_loss   = l1_fn(imcnn, imfull)
        ssim_loss = _ssim_loss(ssim_fn, imcnn, imfull)
        loss = l1_loss + ssim_loss
        if kspace_weight > 0:
            kfull_freq = fft2c_fn(imfull.to(torch.complex64))
            kcnn_freq  = fft2c_fn(imcnn.to(torch.complex64))
            ks_loss = kspace_weight * l1_fn(kcnn_freq.abs(), kfull_freq.abs())
            loss = loss + ks_loss
        if complex_weight > 0:
            kfull = item['kfull'].to(ksub.device)          # (B, T, C, H, W)
            s = sens[:, 0]
            xfull = (torch.conj(s) * IFFT2C()(kfull)).sum(dim=2)
            cx_loss = complex_weight * (net.last_image - xfull).abs().mean()
            loss = loss + cx_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        writer.add_scalar('l1_loss', l1_loss.item(), global_step + step)
        writer.add_scalar('ssim_loss', ssim_loss.item(), global_step + step)
        if kspace_weight > 0:
            writer.add_scalar('freq_loss', ks_loss.item(), global_step + step)
        writer.add_scalar('trn_loss', loss.item(), global_step + step)
        avg_loss = 0.9 * avg_loss + 0.1 * loss.item() \
            if step > 0 else loss.item()
        if (step + 1) % 50 == 0:
            step_time = (time.perf_counter() - start_time_step) / 50
            print(f'[o] Step [{step + 1:4d}/{len(dataloader):4d}] | '
                  f'Loss: {loss.item():.4g} | '
                  f'Avg Loss: {avg_loss:.4g} | '
                  f'Avg Time: {step_time:.2f} s')
            start_time_step = time.perf_counter()
    epoch_time = time.perf_counter() - start_time_epoch
    return avg_loss, epoch_time


def validate(net, dataloader, writer, epoch, use_gpu, ssim_fn,
             r_label='', do_display=False, readout_factor=1):
    """Validate on one dataloader.

    r_label: acceleration factor string appended to TensorBoard scalar names,
             e.g. '4' → logs 'val_ssim_R4', 'val_psnr_R4', 'val_nmse_R4'.
    do_display: whether to write sample images to TensorBoard this call.
    readout_factor: readout crop already applied by the data loader.

    SSIM/PSNR/NMSE are challenge-style, averaged over images (see
    _challenge_metrics); val_ssim_loss is the training SSIM loss.
    Returns (1 - mean challenge SSIM, val_time).
    """
    start_time = time.perf_counter()

    net.eval()
    sum_ssim_loss = 0.0
    metrics = []
    display_imfull, display_imcnn, display_mask = [], [], []

    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            ksub   = item['ksub'].to('cuda')   if use_gpu else item['ksub']
            mask   = item['mask'].to('cuda')   if use_gpu else item['mask']
            imfull = item['imfull'].to('cuda') if use_gpu else item['imfull']
            if 'sens' in item:
                sens = item['sens'].to('cuda') if use_gpu else item['sens']
                imcnn = net(ksub, mask, sens)
                mask = mask.select(-3, 0)        # drop the coil dim
            else:
                imcnn = net(ksub, mask)

            sum_ssim_loss += _ssim_loss(ssim_fn, imcnn, imfull).item()
            metrics += _challenge_metrics(ssim_fn, imcnn, imfull,
                                          readout_factor)

            if do_display and idx % max(len(dataloader) // 16, 1) == 0:
                display_imfull.append(imfull[:1])
                display_imcnn.append(imcnn[:1])
                display_mask.append(mask[:1])

    avg_ssim_loss = sum_ssim_loss / len(dataloader)
    avg_ssim, avg_psnr, avg_nmse = (sum(m[i] for m in metrics) / len(metrics)
                                    for i in range(3))

    suffix = f'_R{r_label}' if r_label else ''
    writer.add_scalar(f'val_ssim_loss{suffix}', avg_ssim_loss, epoch)
    writer.add_scalar(f'val_ssim{suffix}',      avg_ssim,      epoch)
    writer.add_scalar(f'val_psnr{suffix}',      avg_psnr,      epoch)
    writer.add_scalar(f'val_nmse{suffix}',      avg_nmse,      epoch)
    if do_display and display_imfull:
        display(writer, display_imfull, display_imcnn, display_mask, epoch)

    val_time = time.perf_counter() - start_time
    print(f'    [R={r_label}] SSIM {avg_ssim:.4f} | PSNR {avg_psnr:.2f} dB | '
          f'NMSE {avg_nmse:.4f} ({len(metrics)} images)')
    return 1.0 - avg_ssim, val_time


def _select_2d(x):
    """For 5-D (B,C,T,H,W) tensors pick the middle frame; leave 4-D unchanged."""
    if x.dim() == 5:
        return x[:, :, x.shape[2] // 2, :, :]
    return x


def display(writer, display_imfull, display_imcnn, display_mask, epoch):
    display_imfull = [_select_2d(x) for x in display_imfull]
    display_imcnn = [_select_2d(x) for x in display_imcnn]
    display_mask = [_select_2d(x) for x in display_mask]
    max_width = max(imfull.shape[-1] for imfull in display_imfull)
    max_height = max(imfull.shape[-2] for imfull in display_imfull)
    for i in range(len(display_imfull)):
        imfull, imcnn, mask = \
            display_imfull[i], display_imcnn[i], display_mask[i]
        diff_width = max_width - imfull.shape[-1]
        diff_height = max_height - imfull.shape[-2]
        pad = (diff_width // 2, diff_width - diff_width // 2,
               diff_height // 2, diff_height - diff_height // 2)
        display_imfull[i], display_imcnn[i], display_mask[i] = \
            F.pad(imfull, pad), F.pad(imcnn, pad), F.pad(mask, pad)
    display_imfull = torch.cat(display_imfull)
    display_imcnn = torch.cat(display_imcnn)
    display_mask = torch.cat(display_mask)
    writer.add_image('target', make_grid(
        display_imfull, nrow=4, normalize=True, scale_each=True), epoch)
    writer.add_image('output', make_grid(
        display_imcnn, nrow=4, normalize=True, scale_each=True), epoch)
    writer.add_image('mask', make_grid(
        display_mask, nrow=4, normalize=True, scale_each=True), epoch)


def main():
    args = get_parser().parse_args()
    given = lambda flag: any(a == '--' + flag or a.startswith('--' + flag
                                                             + '=')
                             for a in sys.argv[1:])
    if args.stc3:
        # default stC^3-Net: v2 with the ACS-calibrated (SPIRiT) k-net,
        # the best of the 3-subject comparison (2026-10-30)
        args.stc3v2 = True
        if not given('knet_domain'):
            args.knet_domain = 'acs'
        # running statistics made validation depend on the last training
        # samples (VISTA: +0.002-0.012 SSIM and stable epochs without them)
        if not given('norm_stats'):
            args.norm_stats = 'instance'
    if args.stc3v1:
        args.temporal, args.knet = True, 'shallow'
        args.temporal_padding, args.image_only = 'circular', False
        # default sizes (~5.9 M parameters, comparable to DL-ESPIRiT's
        # 5.1 M) and blend DC (soft DC trained unstably and finished
        # lower); options given on the command line take precedence
        for flag, value in {**STC3_SIZES, 'dc': 'blend'}.items():
            if not given(flag):
                setattr(args, flag, value)
    if args.stc3v2:
        args.temporal, args.temporal_padding = True, 'circular'
        for flag, value in {**STC3_SIZES, 'dc': 'grad'}.items():
            if not given(flag):
                setattr(args, flag, value)
        if args.dc not in ('grad', 'cg'):
            raise ValueError('--stc3v2 takes --dc grad or cg.')
    elif args.dc in ('grad', 'cg') or args.complex_weight > 0:
        raise ValueError('--dc grad/cg and --complex_weight need --stc3v2.')
    if args.interleave:
        args.mask = 'interleave'
    trn_dir = args.trn_dir
    val_dir = args.val_dir
    sample_ratio = args.sample_ratio
    real = args.real
    temporal = args.temporal
    multi_coil = 'MultiCoil' in trn_dir
    cascade_depth = args.cascade_depth
    lamda = args.lamda
    net_depth = args.net_depth
    num_filters = args.num_filters
    kernel_size = args.kernel_size
    bias = args.bias
    normalization = args.normalization
    activation = args.activation
    down = args.down
    up = args.up
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    use_gpu = args.use_gpu
    batch_size = args.batch_size
    scheduler_type = args.scheduler
    num_workers = args.num_workers
    kspace_weight = args.freq_weight

    net_name = []
    if temporal:
        net_name.append('temporal')
    if not real:
        net_name.append('complex')
    net_name.append('multi' if multi_coil else 'single')
    if cascade_depth == 0:
        net_name.append('kspace')
    elif cascade_depth == 1:
        net_name.append('image')
    else:
        net_name.append('cascade' + str(100 * lamda)[:2])
    net_name.append('unet')
    net_name.append(str(net_depth))
    net_name.append(str(num_filters))
    net_name.append(str(kernel_size))
    if bias:
        net_name.append('bias')
    net_name.append(normalization)
    net_name.append(activation)
    net_name.append(down)
    net_name.append(up)
    net_name = '_'.join(net_name)
    out_dir = os.path.join(args.out_dir, args.exp_name or net_name)
    log_dir = os.path.join(out_dir, 'log')
    ckpt_filename = os.path.join(out_dir, 'ckpt.pt')
    best_ckpt_filename = os.path.join(out_dir, 'best_ckpt.pt')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Settings needed to rebuild the model later (scripts/preview_recon.py).
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    net = get_net(real, multi_coil, temporal, cascade_depth, lamda,
                  net_depth, num_filters, kernel_size, bias,
                  normalization, activation, down, up,
                  num_coils=args.num_coils if args.coil_mode == 'channel'
                  else None, final_combine=args.final_combine,
                  neighbors=args.neighbors,
                  stc3v2=stc3v2_options(vars(args)))
    apply_model_options(net, vars(args))
    if use_gpu:
        net = net.to('cuda')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    if temporal:
        trn_dataloader  = get_temporal_dataloader(trn_dir, True, sample_ratio,
                                                  batch_size, num_workers,
                                                  args.crop_readout,
                                                  args.train_R, multi_coil,
                                                  args.clip_len,
                                                  pattern=args.mask,
                                                  kt_sigma=args.kt_sigma)
        val_dataloaders = get_temporal_val_dataloaders(val_dir, sample_ratio,
                                                       batch_size, num_workers,
                                                       args.crop_readout,
                                                       args.val_R, multi_coil,
                                                       args.clip_len,
                                                       pattern=args.mask,
                                                       kt_sigma=args.kt_sigma)
    else:
        trn_dataloader  = get_dataloader(trn_dir, True, multi_coil, sample_ratio,
                                         batch_size, num_workers,
                                         args.crop_readout, args.train_R,
                                         neighbors=args.neighbors,
                                         pattern=args.mask,
                                         kt_sigma=args.kt_sigma)
        val_dataloaders = get_val_dataloaders(val_dir, multi_coil, sample_ratio,
                                              batch_size, num_workers,
                                              args.crop_readout, args.val_R,
                                              neighbors=args.neighbors,
                                              pattern=args.mask,
                                              kt_sigma=args.kt_sigma)
    writer = SummaryWriter(log_dir)

    total_parameters, trainable_parameters = count_parameters(net)
    print(f'[o] Total number of parameters: {total_parameters} | '
          f'Number of trainable parameters: {trainable_parameters}')
    device = 'cuda' if use_gpu else 'cpu'
    l1_fn    = torch.nn.L1Loss().to(device)
    ssim_fn  = SSIMLoss().to(device)
    fft2c_fn = FFT2C().to(device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 1e-2) \
        if scheduler_type == 'cosine' else None

    if os.path.exists(ckpt_filename):
        net, optimizer, scheduler, best_val_loss, start_epoch = \
            load_ckpt(ckpt_filename, net, optimizer, scheduler)
        print(f'[o] Resumed from {ckpt_filename} at epoch {start_epoch}.')
    else:
        start_epoch = 0
        best_val_loss = float('Inf')

    for epoch in range(start_epoch, num_epochs):
        trn_loss, trn_time = \
            train(net, optimizer, trn_dataloader, writer, epoch, use_gpu,
                  l1_fn, ssim_fn, fft2c_fn, kspace_weight,
                  args.complex_weight)
        val_ssim_losses, val_time = [], 0.0
        display_R = 8 if 8 in args.val_R else args.val_R[0]
        for r, val_dl in val_dataloaders.items():
            sl, vt = validate(net, val_dl, writer, epoch, use_gpu, ssim_fn,
                              r_label=str(r), do_display=(r == display_R),
                              readout_factor=2 if args.crop_readout else 1)
            val_ssim_losses.append(sl)
            val_time += vt
        val_loss = sum(val_ssim_losses) / len(val_ssim_losses)
        writer.add_scalar('val_loss', val_loss, epoch)
        if scheduler:
            scheduler.step()
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(ckpt_filename, net, optimizer, scheduler,
                      best_val_loss, epoch)
            shutil.copyfile(ckpt_filename, best_ckpt_filename)
        else:
            save_ckpt(ckpt_filename, net, optimizer, scheduler,
                      best_val_loss, epoch)
        lr = scheduler.get_last_lr()[0] if scheduler else learning_rate
        print(f'[o] Epoch [{epoch + 1:3d}/{num_epochs:3d}] | '
              f'LR: {lr:.2e} | '
              f'Training Loss: {trn_loss:.4g} | '
              f'Validation Loss: {val_loss:.4g} | '
              f'Training Time: {trn_time:.2f} s | '
              f'Validation Time: {val_time:.2f} s')

    writer.close()


if __name__ == '__main__':
    main()
