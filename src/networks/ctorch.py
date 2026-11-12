import torch
from torch.nn import *


def _fused_complex_conv(conv_r, conv_i, input, conv_fn):
    """Complex convolution (Wr + iWi) * (xr + ixi) computed as ONE real
    convolution on stacked [xr, xi] channels with the block weight
    [[Wr, -Wi], [Wi, Wr]]: the same arithmetic as the four separate
    convolutions (conv_r(xr) - conv_i(xi)) + i(conv_r(xi) + conv_i(xr)),
    with 1 kernel launch instead of 4 (~1.5x faster for the cascade models).
    """
    wr, wi = conv_r.weight, conv_i.weight
    if conv_r.groups != 1:          # block trick needs groups=1
        return (conv_r(input.real) - conv_i(input.imag)) + \
            1j * (conv_r(input.imag) + conv_i(input.real))
    weight = torch.cat([torch.cat([wr, -wi], 1), torch.cat([wi, wr], 1)], 0)
    bias = None
    if conv_r.bias is not None:
        bias = torch.cat([conv_r.bias - conv_i.bias,
                          conv_r.bias + conv_i.bias])
    x = torch.cat([input.real, input.imag], 1)
    padding = conv_r.padding
    if conv_r.padding_mode != 'zeros':
        x = torch.nn.functional.pad(x, conv_r._reversed_padding_repeated_twice,
                                    mode=conv_r.padding_mode)
        padding = 0
    y = conv_fn(x, weight, bias, conv_r.stride, padding, conv_r.dilation)
    n = wr.shape[0]
    return torch.complex(y[:, :n], y[:, n:])


class CConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, dilation, groups,
                             bias, padding_mode)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, dilation, groups,
                             bias, padding_mode)

    def forward(self, input):
        return _fused_complex_conv(self.conv_r, self.conv_i, input,
                                   torch.nn.functional.conv2d)


class CConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        super().__init__()
        self.conv_transpose_r = \
            ConvTranspose2d(in_channels, out_channels, kernel_size,
                            stride, padding, output_padding, groups,
                            bias, dilation, padding_mode)
        self.conv_transpose_i = \
            ConvTranspose2d(in_channels, out_channels, kernel_size,
                            stride, padding, output_padding, groups,
                            bias, dilation, padding_mode)

    def forward(self, input):
        return (self.conv_transpose_r(input.real) -
                self.conv_transpose_i(input.imag)) + \
            1j * (self.conv_transpose_r(input.imag) +
                  self.conv_transpose_i(input.real))


class CBatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.norm_r = BatchNorm2d(num_features, eps, momentum,
                                  affine, track_running_stats)
        self.norm_i = BatchNorm2d(num_features, eps, momentum,
                                  affine, track_running_stats)

    def forward(self, input):
        return self.norm_r(input.real) + 1j * self.norm_i(input.imag)


class CInstanceNorm2d(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.norm_r = InstanceNorm2d(num_features, eps, momentum,
                                     affine, track_running_stats)
        self.norm_i = InstanceNorm2d(num_features, eps, momentum,
                                     affine, track_running_stats)

    def forward(self, input):
        return self.norm_r(input.real) + 1j * self.norm_i(input.imag)


class CMaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.max_pool = MaxPool2d(kernel_size, stride, padding,
                                  dilation, return_indices=True)

    def forward(self, input):
        output, indices = self.max_pool(torch.abs(input))
        output_shape = output.shape
        output = input.flatten(start_dim=2).gather(
            dim=2, index=indices.flatten(start_dim=2)).reshape(output_shape)
        return output


class CAvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.avg_pool = AvgPool2d(kernel_size, stride, padding)

    def forward(self, input):
        return self.avg_pool(input.real) + 1j * self.avg_pool(input.imag)


class CReLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = ReLU(inplace)

    def forward(self, input):
        return self.relu(input.real) + 1j * self.relu(input.imag)


class CLeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.leaky_relu = LeakyReLU(negative_slope, inplace)

    def forward(self, input):
        return self.leaky_relu(input.real) + 1j * self.leaky_relu(input.imag)


class CUpsamplingBilinear2d(Module):
    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.upsample = UpsamplingBilinear2d(size, scale_factor)

    def forward(self, input):
        return self.upsample(input.real) + 1j * self.upsample(input.imag)


# ── Channel attention (Squeeze-and-Excitation) ───────────────────────────────

class SEBlock(Module):
    """Real-valued channel attention for (B, C, H, W) feature maps."""
    def __init__(self, channels, r=16):
        super().__init__()
        mid = max(channels // r, 4)
        self.pool = AdaptiveAvgPool2d(1)
        self.fc = Sequential(
            Linear(channels, mid), ReLU(),
            Linear(mid, channels), Sigmoid())

    def forward(self, x):
        scale = self.fc(self.pool(x).flatten(1)).view(-1, x.shape[1], 1, 1)
        return x * scale


class CSEBlock(Module):
    """Complex channel attention for (B, C, H, W) complex feature maps.

    Gates on magnitude so the phase is preserved unchanged.
    """
    def __init__(self, channels, r=16):
        super().__init__()
        mid = max(channels // r, 4)
        self.pool = AdaptiveAvgPool2d(1)
        self.fc = Sequential(
            Linear(channels, mid), ReLU(),
            Linear(mid, channels), Sigmoid())

    def forward(self, x):
        scale = self.fc(self.pool(x.abs()).flatten(1)).view(-1, x.shape[1], 1, 1)
        return x * scale


class CSEBlock3d(Module):
    """Complex channel attention for (B, C, T, H, W) complex feature maps."""
    def __init__(self, channels, r=16):
        super().__init__()
        mid = max(channels // r, 4)
        self.pool = AdaptiveAvgPool3d(1)
        self.fc = Sequential(
            Linear(channels, mid), ReLU(),
            Linear(mid, channels), Sigmoid())

    def forward(self, x):
        scale = self.fc(self.pool(x.abs()).flatten(1)).view(-1, x.shape[1], 1, 1, 1)
        return x * scale


# ── 3-D complex layers ────────────────────────────────────────────────────────

class CConv3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__()
        self.conv_r = Conv3d(in_channels, out_channels, kernel_size,
                             stride, padding, dilation, groups,
                             bias, padding_mode)
        self.conv_i = Conv3d(in_channels, out_channels, kernel_size,
                             stride, padding, dilation, groups,
                             bias, padding_mode)

    def forward(self, input):
        return _fused_complex_conv(self.conv_r, self.conv_i, input,
                                   torch.nn.functional.conv3d)


class CConvTranspose3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        super().__init__()
        self.conv_transpose_r = \
            ConvTranspose3d(in_channels, out_channels, kernel_size,
                            stride, padding, output_padding, groups,
                            bias, dilation, padding_mode)
        self.conv_transpose_i = \
            ConvTranspose3d(in_channels, out_channels, kernel_size,
                            stride, padding, output_padding, groups,
                            bias, dilation, padding_mode)

    def forward(self, input):
        return (self.conv_transpose_r(input.real) -
                self.conv_transpose_i(input.imag)) + \
            1j * (self.conv_transpose_r(input.imag) +
                  self.conv_transpose_i(input.real))


class CAvgPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.avg_pool = AvgPool3d(kernel_size, stride, padding)

    def forward(self, input):
        return self.avg_pool(input.real) + 1j * self.avg_pool(input.imag)


class CMaxPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.max_pool = MaxPool3d(kernel_size, stride, padding,
                                  dilation, return_indices=True)

    def forward(self, input):
        output, indices = self.max_pool(torch.abs(input))
        output_shape = output.shape
        output = input.flatten(start_dim=2).gather(
            dim=2, index=indices.flatten(start_dim=2)).reshape(output_shape)
        return output


class CBatchNorm3d(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.norm_r = BatchNorm3d(num_features, eps, momentum,
                                  affine, track_running_stats)
        self.norm_i = BatchNorm3d(num_features, eps, momentum,
                                  affine, track_running_stats)

    def forward(self, input):
        return self.norm_r(input.real) + 1j * self.norm_i(input.imag)


class CInstanceNorm3d(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.norm_r = InstanceNorm3d(num_features, eps, momentum,
                                     affine, track_running_stats)
        self.norm_i = InstanceNorm3d(num_features, eps, momentum,
                                     affine, track_running_stats)

    def forward(self, input):
        return self.norm_r(input.real) + 1j * self.norm_i(input.imag)


class CUpsamplingTrilinear3d(Module):
    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.upsample = Upsample(size=size, scale_factor=scale_factor,
                                 mode='trilinear', align_corners=True)

    def forward(self, input):
        return self.upsample(input.real) + 1j * self.upsample(input.imag)


class CConv2plus1d(Module):
    """Factored (2+1)D complex conv: spatial (1,k,k) → norm → act → temporal (k,1,1).

    The intermediate nonlinearity makes this strictly more expressive than a
    plain 3D conv with the same parameter budget.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        self.spatial = CConv3d(
            in_channels, out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, kernel_size // 2, kernel_size // 2), bias=bias)
        self.norm = CInstanceNorm3d(out_channels)
        self.act = CReLU()
        self.temporal = CConv3d(
            out_channels, out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0), bias=bias)

    def forward(self, input):
        return self.temporal(self.act(self.norm(self.spatial(input))))
