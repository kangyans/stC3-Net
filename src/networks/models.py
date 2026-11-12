from .dc import *
from .unet import *


def _run_knet(knet, k, coil_dim, coils_as_channels, real):
    """Apply a k-space net to multi-coil k-space k: (B, 1, C, H, W) complex.

    coils_as_channels: one pass with all coils as input channels (the net
    can use cross-coil, GRAPPA-like correlations). Otherwise every coil is
    processed independently as its own batch element.
    real: the net is a two-channel real UNet (real/imag stacked as channels).
    """
    B, C = k.shape[0], k.shape[coil_dim]
    if coils_as_channels:
        x = k.squeeze(1)                                    # (B, C, H, W)
    else:
        x = k.transpose(1, coil_dim).reshape(B * C, 1, *k.shape[-2:])
    n = x.shape[1]
    if real:
        y = knet(torch.cat([x.real, x.imag], dim=1))
        y = torch.complex(y[:, :n].contiguous(), y[:, n:].contiguous())
    else:
        y = knet(x)
    if coils_as_channels:
        return y.unsqueeze(1)
    return y.reshape(B, C, *y.shape[1:]).transpose(1, coil_dim)


def _final_image(kcnn, sens, coil_dim, final_combine):
    """Magnitude image from multi-coil k-space kcnn: (B, 1, C, H, W).

    'rss': root-sum-of-squares of the coil images, the same combination as
           the challenge's (and TargetRecon's) reference images.
    'sense': |sum_c conj(S_c) x_c|, the original behaviour; it differs from
           the RSS target by a few percent even for perfect k-space.
    """
    coil_images = IFFT2C()(kcnn)
    if final_combine == 'rss':
        return torch.linalg.vector_norm(coil_images, dim=coil_dim)
    return torch.abs(CoilCombine(coil_dim)(coil_images, sens))


class SingleCoilKspaceDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=64, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', channel_dim=1):
        super().__init__()
        self.knet = UNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.real_complex = RealComplex(channel_dim)
        self.complex_real = ComplexReal(channel_dim)
        self.ifft2c = IFFT2C()

    def forward(self, *args):
        ksub = args[0]
        kcnn = self.real_complex(self.knet(self.complex_real(ksub)))
        imcnn = self.ifft2c(kcnn)
        return torch.abs(imcnn)


class MultiCoilKspaceDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=64, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', channel_dim=1, coil_dim=2,
                 num_coils=None,
                 final_combine='rss'):
        super().__init__()
        k = num_coils or 1
        self.knet = UNet(
            in_channels * k, out_channels * k, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.ifft2c = IFFT2C()
        self.coil_combine = CoilCombine(coil_dim)
        self.coil_dim = coil_dim
        self.num_coils = num_coils
        self.final_combine = final_combine

    def forward(self, *args):
        ksub, sens = args[0], args[2]
        kcnn = _run_knet(self.knet, ksub, self.coil_dim,
                         bool(self.num_coils), real=True)
        return _final_image(kcnn, sens, self.coil_dim, self.final_combine)


class SingleCoilComplexKspaceDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear'):
        super().__init__()
        self.knet = CUNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.ifft2c = IFFT2C()

    def forward(self, *args):
        ksub = args[0]
        kcnn = self.knet(ksub)
        imcnn = self.ifft2c(kcnn)
        return torch.abs(imcnn)


class MultiCoilComplexKspaceDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', coil_dim=2, num_coils=None,
                 final_combine='rss'):
        super().__init__()
        k = num_coils or 1
        self.knet = CUNet(
            in_channels * k, out_channels * k, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.ifft2c = IFFT2C()
        self.coil_combine = CoilCombine(coil_dim)
        self.coil_dim = coil_dim
        self.num_coils = num_coils
        self.final_combine = final_combine

    def forward(self, *args):
        ksub, sens = args[0], args[2]
        kcnn = _run_knet(self.knet, ksub, self.coil_dim,
                         bool(self.num_coils), real=False)
        return _final_image(kcnn, sens, self.coil_dim, self.final_combine)


class SingleCoilImageDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=64, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', channel_dim=1):
        super().__init__()
        self.inet = UNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.real_complex = RealComplex(channel_dim)
        self.complex_real = ComplexReal(channel_dim)
        self.ifft2c = IFFT2C()

    def forward(self, *args):
        ksub = args[0]
        imsub = self.ifft2c(ksub)
        imcnn = self.real_complex(self.inet(self.complex_real(imsub)))
        return torch.abs(imcnn)


class MultiCoilImageDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=64, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', channel_dim=1, coil_dim=2):
        super().__init__()
        self.inet = UNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.real_complex = RealComplex(channel_dim)
        self.complex_real = ComplexReal(channel_dim)
        self.ifft2c = IFFT2C()
        self.coil_combine = CoilCombine(coil_dim)

    def forward(self, *args):
        ksub, sens = args[0], args[2]
        imsub = self.coil_combine(self.ifft2c(ksub), sens)
        imcnn = self.real_complex(self.inet(self.complex_real(imsub)))
        return torch.abs(imcnn)


class SingleCoilComplexImageDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear'):
        super().__init__()
        self.inet = CUNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.ifft2c = IFFT2C()

    def forward(self, *args):
        ksub = args[0]
        imsub = self.ifft2c(ksub)
        imcnn = self.inet(imsub)
        return torch.abs(imcnn)


class MultiCoilComplexImageDomainNet(Module):
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', coil_dim=2):
        super().__init__()
        self.inet = CUNet(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.ifft2c = IFFT2C()
        self.coil_combine = CoilCombine(coil_dim)

    def forward(self, *args):
        ksub, sens = args[0], args[2]
        imsub = self.coil_combine(self.ifft2c(ksub), sens)
        imcnn = self.inet(imsub)
        return torch.abs(imcnn)


class SingleCoilCascadeCrossDomainNet(Module):
    def __init__(self, in_channels, out_channels, cascade_depth=2,
                 net_depth=4, num_filters=64, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg',
                 up='bilinear', lamda=0.99, channel_dim=1):
        super().__init__()
        self.knets = ModuleList([UNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.inets = ModuleList([UNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.real_complex = RealComplex(channel_dim)
        self.complex_real = ComplexReal(channel_dim)
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.data_consistency = DataConsistency(lamda)

    def forward(self, *args):
        ksub, mask = args[0], args[1]
        kcnn = ksub
        for i, (knet, inet) in enumerate(zip(self.knets, self.inets)):
            kin = kcnn
            kcnn = self.real_complex(knet(self.complex_real(kcnn)))
            imcnn = self.ifft2c(kcnn)
            imcnn = self.real_complex(inet(self.complex_real(imcnn)))
            kcnn = self.fft2c(imcnn)
            kcnn = self.data_consistency(kcnn, mask, ksub, kin, i)
        imcnn = self.ifft2c(kcnn)
        return torch.abs(imcnn)


class MultiCoilCascadeCrossDomainNet(Module):
    def __init__(self, in_channels, out_channels, cascade_depth=2,
                 net_depth=4, num_filters=64, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg',
                 up='bilinear', lamda=0.99, channel_dim=1, coil_dim=2,
                 num_coils=None,
                 final_combine='rss'):
        super().__init__()
        k = num_coils or 1
        self.knets = ModuleList([UNet(
            in_channels * k, out_channels * k, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.inets = ModuleList([UNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.real_complex = RealComplex(channel_dim)
        self.complex_real = ComplexReal(channel_dim)
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.coil_split = CoilSplit(coil_dim)
        self.coil_combine = CoilCombine(coil_dim)
        self.data_consistency = DataConsistency(lamda)
        self.coil_dim = coil_dim
        self.num_coils = num_coils
        self.final_combine = final_combine

    def forward(self, *args):
        ksub, mask, sens = args[0], args[1], args[2]
        kcnn = ksub
        for i, (knet, inet) in enumerate(zip(self.knets, self.inets)):
            kin = kcnn
            kcnn = _run_knet(knet, kcnn, self.coil_dim,
                             bool(self.num_coils), real=True)
            imcnn = self.coil_combine(self.ifft2c(kcnn), sens)
            imcnn = self.real_complex(inet(self.complex_real(imcnn)))
            kcnn = self.fft2c(self.coil_split(imcnn, sens))
            kcnn = self.data_consistency(kcnn, mask, ksub, kin, i)
        return _final_image(kcnn, sens, self.coil_dim, self.final_combine)


class SingleCoilComplexCascadeCrossDomainNet(Module):
    def __init__(self, in_channels, out_channels, cascade_depth=2,
                 net_depth=4, num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg',
                 up='bilinear', lamda=0.99):
        super().__init__()
        self.knets = ModuleList([CUNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.inets = ModuleList([CUNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.data_consistency = DataConsistency(lamda)

    def forward(self, *args):
        ksub, mask = args[0], args[1]
        kcnn = ksub
        for i, (knet, inet) in enumerate(zip(self.knets, self.inets)):
            kin = kcnn
            kcnn = knet(kcnn)
            imcnn = self.ifft2c(kcnn)
            imcnn = inet(imcnn)
            kcnn = self.fft2c(imcnn)
            kcnn = self.data_consistency(kcnn, mask, ksub, kin, i)
        imcnn = self.ifft2c(kcnn)
        return torch.abs(imcnn)


class MultiCoilComplexCascadeCrossDomainNet(Module):
    def __init__(self, in_channels, out_channels, cascade_depth=2,
                 net_depth=4, num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg',
                 up='bilinear', lamda=0.99, coil_dim=2, num_coils=None,
                 final_combine='rss'):
        super().__init__()
        k = num_coils or 1
        self.knets = ModuleList([CUNet(
            in_channels * k, out_channels * k, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.inets = ModuleList([CUNet(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.coil_split = CoilSplit(coil_dim)
        self.coil_combine = CoilCombine(coil_dim)
        self.data_consistency = DataConsistency(lamda)
        self.coil_dim = coil_dim
        self.num_coils = num_coils
        self.final_combine = final_combine

    def forward(self, *args):
        ksub, mask, sens = args[0], args[1], args[2]
        kcnn = ksub
        for i, (knet, inet) in enumerate(zip(self.knets, self.inets)):
            kin = kcnn
            kcnn = _run_knet(knet, kcnn, self.coil_dim,
                             bool(self.num_coils), real=False)
            imcnn = self.coil_combine(self.ifft2c(kcnn), sens)
            imcnn = inet(imcnn)
            kcnn = self.fft2c(self.coil_split(imcnn, sens))
            kcnn = self.data_consistency(kcnn, mask, ksub, kin, i)
        return _final_image(kcnn, sens, self.coil_dim, self.final_combine)


# ── Temporal (2+1)D models ────────────────────────────────────────────────────
# These models load all frames of a slice together as (B, 1, T, H, W) tensors.

class SingleCoilComplexTemporalImageDomainNet(Module):
    """Image-domain-only model with (2+1)D temporal convolutions (single coil).

    Input ksub: (B, 1, T, kH, kW) complex — one channel, T frames.
    Output:     (B, 1, T, H, W)  real   — magnitude image per frame.
    """
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear'):
        super().__init__()
        self.inet = CUNet2p1d(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.ifft2c = IFFT2C()

    def forward(self, *args):
        ksub = args[0]
        imsub = self.ifft2c(ksub)
        imcnn = self.inet(imsub)
        return torch.abs(imcnn)


class SingleCoilComplexTemporalCascadeCrossDomainNet(Module):
    """Cascade cross-domain model with (2+1)D temporal convolutions (single coil).

    Input ksub: (B, 1, T, kH, kW) complex.
    Input mask: (B, 1, T, kH, kW) float.
    Output:     (B, 1, T, H, W)  real.
    """
    def __init__(self, in_channels, out_channels, cascade_depth=2,
                 net_depth=4, num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', lamda=0.99):
        super().__init__()
        self.knets = ModuleList([CUNet2p1d(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.inets = ModuleList([CUNet2p1d(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.data_consistency = DataConsistency(lamda)

    def forward(self, *args):
        ksub, mask = args[0], args[1]
        kcnn = ksub
        for i, (knet, inet) in enumerate(zip(self.knets, self.inets)):
            kin = kcnn
            kcnn = knet(kcnn)
            imcnn = self.ifft2c(kcnn)
            imcnn = inet(imcnn)
            kcnn = self.fft2c(imcnn)
            kcnn = self.data_consistency(kcnn, mask, ksub, kin, i)
        imcnn = self.ifft2c(kcnn)
        return torch.abs(imcnn)


class MultiCoilComplexTemporalImageDomainNet(Module):
    """Image-domain-only (2+1)D model on the coil-combined image (multi-coil).

    Input ksub, mask, sens: (B, 1, T, C, kH, kW). Output: (B, 1, T, H, W).
    """
    def __init__(self, in_channels, out_channels, net_depth=4,
                 num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear'):
        super().__init__()
        self.inet = CUNet2p1d(
            in_channels, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
        self.ifft2c = IFFT2C()
        self.coil_combine = CoilCombine(3)

    def forward(self, *args):
        ksub, sens = args[0], args[2]
        imsub = self.coil_combine(self.ifft2c(ksub), sens)    # (B, 1, T, H, W)
        return torch.abs(self.inet(imsub))


class MultiCoilComplexTemporalCascadeCrossDomainNet(Module):
    """Cascade cross-domain model with (2+1)D convolutions (multi-coil).

    Input ksub, mask, sens: (B, 1, T, C, kH, kW). Output: (B, 1, T, H, W).
    The k-net sees all coils (as channels) and all frames of a slice at once;
    between the nets the coils are combined into one complex image per frame
    and the i-net works on the frame sequence.
    """
    def __init__(self, in_channels, out_channels, cascade_depth=2,
                 net_depth=4, num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu',
                 down='avg', up='bilinear', lamda=0.99, num_coils=10,
                 final_combine='rss'):
        super().__init__()
        self.knets = ModuleList([CUNet2p1d(
            in_channels * num_coils, out_channels * num_coils, net_depth,
            num_filters, kernel_size, bias, normalization, activation,
            down, up) for _ in range(cascade_depth)])
        self.inets = ModuleList([CUNet2p1d(
            in_channels, out_channels, net_depth, num_filters, kernel_size,
            bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.coil_dim = 3
        self.coil_split = CoilSplit(self.coil_dim)
        self.coil_combine = CoilCombine(self.coil_dim)
        self.data_consistency = DataConsistency(lamda)
        self.final_combine = final_combine

    def forward(self, *args):
        ksub, mask, sens = args[0], args[1], args[2]
        kcnn = ksub
        for i, (knet, inet) in enumerate(zip(self.knets, self.inets)):
            kin = kcnn
            x = kcnn[:, 0].transpose(1, 2)                  # (B, C, T, kH, kW)
            if getattr(self, 'knet_mask', False):           # sampled lines
                x = torch.cat([x, mask[:, 0, :, :1].transpose(1, 2)
                               .to(x.dtype)], dim=1)
            x = knet(x)
            kcnn = x.transpose(1, 2).unsqueeze(1)
            imcnn = self.coil_combine(self.ifft2c(kcnn), sens)
            imcnn = inet(imcnn)                             # (B, 1, T, H, W)
            kcnn = self.fft2c(self.coil_split(imcnn, sens))
            kcnn = self.data_consistency(kcnn, mask, ksub, kin, i)
        return _final_image(kcnn, sens, self.coil_dim, self.final_combine)


class MultiCoilComplex25DCascadeNet(Module):
    """2.5D cascade: model A's 2D cross-domain cascade on the centre frame,
    with the 2n neighbouring frames (zero-filled) as extra input channels.

    Input ksub, mask, sens: (B, 1, F, C, kH, kW) with F = 2n + 1 frames, the
    centre one being reconstructed. Output: (B, 1, H, W). The k-net sees the
    coils of all F frames (F*C channels -> C), the i-net the F coil-combined
    images (F channels -> 1, the centre one being the current estimate).
    Residual UNets add only the centre frame, so at initialisation this is
    exactly the 2D model.
    """
    def __init__(self, in_channels, out_channels, cascade_depth=2,
                 net_depth=4, num_filters=32, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg',
                 up='bilinear', lamda=0.99, num_coils=10, num_frames=3,
                 final_combine='rss'):
        super().__init__()
        self.knets = ModuleList([CUNet(
            in_channels * num_coils * num_frames, out_channels * num_coils,
            net_depth, num_filters, kernel_size, bias, normalization,
            activation, down, up) for _ in range(cascade_depth)])
        self.inets = ModuleList([CUNet(
            in_channels * num_frames, out_channels, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up)
            for _ in range(cascade_depth)])
        self.fft2c = FFT2C()
        self.ifft2c = IFFT2C()
        self.coil_split = CoilSplit(2)
        self.coil_combine = CoilCombine(2)
        self.data_consistency = DataConsistency(lamda)
        self.final_combine = final_combine

    def forward(self, *args):
        ksub, mask, sens = args[0], args[1], args[2]
        c = ksub.shape[2] // 2
        k0, m0, s0 = ksub[:, :, c], mask[:, :, c], sens[:, :, c]
        kctx = ksub[:, 0]                                   # (B, F, C, kH, kW)
        ictx = CoilCombine(3)(self.ifft2c(ksub), sens)[:, 0]   # (B, F, H, W)
        kcnn = k0
        for i, (knet, inet) in enumerate(zip(self.knets, self.inets)):
            kin = kcnn
            x = torch.cat([kctx[:, :c], kcnn, kctx[:, c + 1:]], dim=1)
            kcnn = knet(x.flatten(1, 2)).unsqueeze(1)       # (B, 1, C, kH, kW)
            imcnn = self.coil_combine(self.ifft2c(kcnn), s0)   # (B, 1, H, W)
            imcnn = inet(torch.cat([ictx[:, :c], imcnn, ictx[:, c + 1:]], 1))
            kcnn = self.fft2c(self.coil_split(imcnn, s0))
            kcnn = self.data_consistency(kcnn, m0, k0, kin, i)
        return _final_image(kcnn, s0, 2, self.final_combine)


def set_dc_mode(net, mode):
    """Replace a cascade model's data consistency (see DataConsistency)."""
    if mode != 'blend' and hasattr(net, 'data_consistency'):
        net.data_consistency = DataConsistency(
            0.0, mode, num_cascades=len(net.knets))
    return net


def remove_se(net):
    """Drop the squeeze-and-excitation blocks from every UNet (ablation)."""
    for m in net.modules():
        for name, child in list(m.named_children()):
            if isinstance(child, (SEBlock, CSEBlock, CSEBlock3d)):
                setattr(m, name, Identity())
    return net


def make_image_only(net):
    """Drop the k-space nets of a cross-domain cascade, leaving image-domain
    cascades with data consistency (as in E2E-VarNet)."""
    if isinstance(net, MultiCoilComplex25DCascadeNet):
        raise ValueError('--image_only is not supported for 2.5D models.')
    if hasattr(net, 'knets'):
        net.knets = ModuleList([Identity() for _ in net.knets])
    return net


class ShallowCKNet2p1d(Module):
    """k-space net without pooling: `layers` complex (2+1)D convs (3x3 in
    k-space -> norm -> CReLU -> `t_kernel` taps along time, circular),
    full resolution, residual with a zero-initialised last 1x1x1 conv. Local
    k-t kernels as in k-t GRAPPA; t_kernel=1 gives the same net with 2D
    (per-frame) kernels. Input (B, C_in, T, kH, kW), output C_out channels;
    extra input channels (the mask) are not part of the residual.
    linear: no normalisation or activation, i.e. one learned linear k-t
    interpolation kernel (GRAPPA-like)."""
    def __init__(self, in_channels, out_channels, width=32, layers=5,
                 t_kernel=3, linear=False):
        super().__init__()
        blocks, ch = [], in_channels
        for _ in range(layers):
            blocks += [CConv3d(ch, width, (1, 3, 3), padding=(0, 1, 1),
                               bias=False)]
            if not linear:
                blocks += [CInstanceNorm3d(width), CReLU()]
            blocks += [CConv3d(width, width, (t_kernel, 1, 1),
                               padding=(t_kernel // 2, 0, 0), bias=False)]
            ch = width
        for m in blocks:
            if isinstance(m, CConv3d) and t_kernel > 1 and \
                    m.conv_r.kernel_size[0] > 1:
                m.conv_r.padding_mode = m.conv_i.padding_mode = 'circular'
        self.body = Sequential(*blocks)
        self.last = CConv3d(width, out_channels, 1, padding=0, bias=False)
        for p in self.last.parameters():
            torch.nn.init.zeros_(p)
        self.out_channels = out_channels
        self.grad_checkpoint = False

    def _body(self, x):
        return self.last(self.body(x))

    def forward(self, x):
        if self.grad_checkpoint and torch.is_grad_enabled():
            y = checkpoint(self._body, x, use_reentrant=False)
        else:
            y = self._body(x)
        return x[:, :self.out_channels] + y


def set_knet(net, kind='unet', width=32, layers=5, t_kernel=3, mask=False):
    """k-net variants for the (2+1)D cross-domain cascade. kind 'shallow':
    ShallowCKNet2p1d instead of the (2+1)D UNet; mask: the sampling mask is
    an extra input channel. Call before make_residual."""
    if not isinstance(net, MultiCoilComplexTemporalCascadeCrossDomainNet):
        if kind != 'unet' or mask:
            raise ValueError('k-net options need the temporal multi-coil '
                             'cross-domain cascade (--temporal).')
        return net
    net.knet_mask = mask
    if kind == 'unet' and not mask:
        return net
    knets = []
    for k in net.knets:
        c_out = k.last.conv_r.out_channels
        c_in = c_out + int(mask)
        if kind == 'shallow':
            knets.append(ShallowCKNet2p1d(c_in, c_out, width, layers,
                                          t_kernel))
        else:
            first = k.first.block[0].spatial.conv_r
            k.first.block[0].spatial = CConv3d(
                c_in, first.out_channels, first.kernel_size,
                padding=first.padding, bias=first.bias is not None)
            sc = k.first.shortcut
            if not isinstance(sc, Identity):
                k.first.shortcut = CConv3d(c_in, sc.conv_r.out_channels, 1,
                                           padding=0, bias=sc.conv_r.bias
                                           is not None)
            knets.append(k)
    net.knets = ModuleList(knets)
    return net
