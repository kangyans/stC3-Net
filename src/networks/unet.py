from .ctorch import *
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class _DoubleConvBlock(Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, bias=False, normalization='instance',
                 activation='relu'):
        super().__init__()
        if normalization != 'instance' and normalization != 'batch':
            raise ValueError('normalization should be or instance or batch.')
        if activation != 'relu' and activation != 'leakyrelu':
            raise ValueError('activation should be relu or leakyrelu.')
        if not mid_channels:
            mid_channels = out_channels
        self.block = Sequential(
            Conv2d(in_channels, mid_channels, kernel_size,
                   padding='same', bias=bias),
            InstanceNorm2d(mid_channels) if normalization == 'instance'
            else BatchNorm2d(mid_channels),
            ReLU() if activation == 'relu'
            else LeakyReLU(negative_slope=0.1),
            Conv2d(mid_channels, out_channels, kernel_size,
                   padding='same', bias=bias),
            InstanceNorm2d(out_channels) if normalization == 'instance'
            else BatchNorm2d(out_channels),
            ReLU() if activation == 'relu'
            else LeakyReLU(negative_slope=0.1),
            SEBlock(out_channels))
        self.shortcut = Identity() if in_channels == out_channels \
            else Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, input):
        return self.block(input) + self.shortcut(input)


class _CDoubleConvBlock(Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, bias=False, normalization='instance',
                 activation='relu'):
        super().__init__()
        if normalization != 'instance' and normalization != 'batch':
            raise ValueError('normalization should be or instance or batch.')
        if activation != 'relu' and activation != 'leakyrelu':
            raise ValueError('activation should be relu or leakyrelu.')
        if not mid_channels:
            mid_channels = out_channels
        self.block = Sequential(
            CConv2d(in_channels, mid_channels, kernel_size,
                    padding='same', bias=bias),
            CInstanceNorm2d(mid_channels) if normalization == 'instance'
            else CBatchNorm2d(mid_channels),
            CReLU() if activation == 'relu'
            else CLeakyReLU(negative_slope=0.1),
            CConv2d(mid_channels, out_channels, kernel_size,
                    padding='same', bias=bias),
            CInstanceNorm2d(out_channels) if normalization == 'instance'
            else CBatchNorm2d(out_channels),
            CReLU() if activation == 'relu'
            else CLeakyReLU(negative_slope=0.1),
            CSEBlock(out_channels))
        self.shortcut = Identity() if in_channels == out_channels \
            else CConv2d(in_channels, out_channels, kernel_size=1,
                         padding=0, bias=bias)

    def forward(self, input):
        return self.block(input) + self.shortcut(input)


class _DownBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg'):
        super().__init__()
        if down != 'avg' and down != 'max':
            raise ValueError('down should be avg or max.')
        self.down_sample = AvgPool2d(kernel_size=2) if down == 'avg' \
            else MaxPool2d(kernel_size=2)
        self.down_conv = _DoubleConvBlock(
            in_channels, out_channels, None,
            kernel_size, bias, normalization, activation)

    def forward(self, input):
        return self.down_conv(self.down_sample(input))


class _CDownBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg'):
        super().__init__()
        if down != 'avg' and down != 'max':
            raise ValueError('down should be avg or max.')
        self.down_sample = CAvgPool2d(kernel_size=2) if down == 'avg' \
            else CMaxPool2d(kernel_size=2)
        self.down_conv = _CDoubleConvBlock(
            in_channels, out_channels, None,
            kernel_size, bias, normalization, activation)

    def forward(self, input):
        return self.down_conv(self.down_sample(input))


class _UpBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', up='bilinear'):
        super().__init__()
        if up != 'bilinear' and up != 'convtran':
            raise ValueError('up should be bilinear or convtran.')
        if up == 'bilinear':
            self.up_sample = UpsamplingBilinear2d(scale_factor=2)
            self.up_conv = _DoubleConvBlock(
                in_channels, out_channels, in_channels // 2,
                kernel_size, bias, normalization, activation)
        else:
            self.up_sample = ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2, bias=bias)
            self.up_conv = _DoubleConvBlock(
                in_channels, out_channels, None,
                kernel_size, bias, normalization, activation)

    def forward(self, input, skip_connection):
        input = self.up_sample(input)
        diffw = skip_connection.shape[-1] - input.shape[-1]
        diffh = skip_connection.shape[-2] - input.shape[-2]
        input = F.pad(input, (diffw // 2, diffw - diffw // 2,
                              diffh // 2, diffh - diffh // 2))
        return self.up_conv(torch.cat((skip_connection, input), dim=1))


class _CUpBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', up='bilinear'):
        super().__init__()
        if up != 'bilinear' and up != 'convtran':
            raise ValueError('up should be bilinear or convtran.')
        if up == 'bilinear':
            self.up_sample = CUpsamplingBilinear2d(scale_factor=2)
            self.up_conv = _CDoubleConvBlock(
                in_channels, out_channels, in_channels // 2,
                kernel_size, bias, normalization, activation)
        else:
            self.up_sample = CConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2, bias=bias)
            self.up_conv = _CDoubleConvBlock(
                in_channels, out_channels, None,
                kernel_size, bias, normalization, activation)

    def forward(self, input, skip_connection):
        input = self.up_sample(input)
        diffw = skip_connection.shape[-1] - input.shape[-1]
        diffh = skip_connection.shape[-2] - input.shape[-2]
        input = F.pad(input, (diffw // 2, diffw - diffw // 2,
                              diffh // 2, diffh - diffh // 2))
        return self.up_conv(torch.cat((skip_connection, input), dim=1))


class UNet(Module):
    def __init__(self, in_channels, out_channels, depth=4, num_filters=64,
                 kernel_size=3, bias=False, normalization='instance',
                 activation='relu', down='avg', up='bilinear'):
        super().__init__()
        self.residual = False   # see make_residual()
        self.film = None        # see make_conditional()
        self.cond = None
        self.grad_checkpoint = False   # see set_grad_checkpoint()
        factor = 2 if up == 'bilinear' else 1
        self.first = _DoubleConvBlock(
            in_channels, num_filters, None,
            kernel_size, bias, normalization, activation)
        self.downs = ModuleList([_DownBlock(
            num_filters, num_filters * 2, kernel_size,
            bias, normalization, activation, down)])
        num_filters *= 2
        for _ in range(1, depth - 1):
            self.downs.append(_DownBlock(
                num_filters, num_filters * 2, kernel_size,
                bias, normalization, activation, down))
            num_filters *= 2
        self.downs.append(_DownBlock(
            num_filters, num_filters * 2 // factor, kernel_size,
            bias, normalization, activation, down))
        self.ups = ModuleList()
        for _ in range(depth - 1):
            self.ups.append(_UpBlock(
                num_filters * 2, num_filters // factor, kernel_size,
                bias, normalization, activation, up))
            num_filters //= 2
        self.ups.append(_UpBlock(
            num_filters * 2, num_filters, kernel_size,
            bias, normalization, activation, up))
        self.last = Conv2d(num_filters, out_channels, kernel_size=1,
                           padding='same', bias=bias)

    def forward(self, input):
        if self.grad_checkpoint and torch.is_grad_enabled():
            output = checkpoint(self._body, input, use_reentrant=False)
        else:
            output = self._body(input)
        return _residual_input(input, output) + output if self.residual \
            else output

    def _body(self, input):
        output = self.first(input)
        if self.film is not None and self.cond is not None:
            output = output * _film_scale(self.film, self.cond, output)
        skip_connections = [output]
        for down in self.downs[:-1]:
            output = down(output)
            skip_connections.append(output)
        output = self.downs[-1](output)
        for up in self.ups:
            output = up(output, skip_connections.pop())
        return self.last(output)


class CUNet(Module):
    def __init__(self, in_channels, out_channels, depth=4, num_filters=32,
                 kernel_size=3, bias=False, normalization='instance',
                 activation='relu', down='avg', up='bilinear'):
        super().__init__()
        self.residual = False   # see make_residual()
        self.film = None        # see make_conditional()
        self.cond = None
        self.grad_checkpoint = False   # see set_grad_checkpoint()
        factor = 2 if up == 'bilinear' else 1
        self.first = _CDoubleConvBlock(
            in_channels, num_filters, None,
            kernel_size, bias, normalization, activation)
        self.downs = ModuleList([_CDownBlock(
            num_filters, num_filters * 2, kernel_size,
            bias, normalization, activation, down)])
        num_filters *= 2
        for _ in range(1, depth - 1):
            self.downs.append(_CDownBlock(
                num_filters, num_filters * 2, kernel_size,
                bias, normalization, activation, down))
            num_filters *= 2
        self.downs.append(_CDownBlock(
            num_filters, num_filters * 2 // factor, kernel_size,
            bias, normalization, activation, down))
        self.ups = ModuleList()
        for _ in range(depth - 1):
            self.ups.append(_CUpBlock(
                num_filters * 2, num_filters // factor, kernel_size,
                bias, normalization, activation, up))
            num_filters //= 2
        self.ups.append(_CUpBlock(
            num_filters * 2, num_filters, kernel_size,
            bias, normalization, activation, up))
        self.last = CConv2d(num_filters, out_channels, kernel_size=1,
                            padding='same', bias=bias)

    def forward(self, input):
        if self.grad_checkpoint and torch.is_grad_enabled():
            output = checkpoint(self._body, input, use_reentrant=False)
        else:
            output = self._body(input)
        return _residual_input(input, output) + output if self.residual \
            else output

    def _body(self, input):
        output = self.first(input)
        if self.film is not None and self.cond is not None:
            output = output * _film_scale(self.film, self.cond, output)
        skip_connections = [output]
        for down in self.downs[:-1]:
            output = down(output)
            skip_connections.append(output)
        output = self.downs[-1](output)
        for up in self.ups:
            output = up(output, skip_connections.pop())
        return self.last(output)


# ── (2+1)D complex UNet ───────────────────────────────────────────────────────
# Input/output tensors are 5-D: (B, C, T, H, W).
# Pooling is spatial-only — temporal resolution is preserved throughout.

class _CDoubleConv2p1dBlock(Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, bias=False, normalization='instance',
                 activation='relu'):
        super().__init__()
        if normalization not in ('instance', 'batch'):
            raise ValueError('normalization should be instance or batch.')
        if activation not in ('relu', 'leakyrelu'):
            raise ValueError('activation should be relu or leakyrelu.')
        if not mid_channels:
            mid_channels = out_channels
        norm3d = CInstanceNorm3d if normalization == 'instance' else CBatchNorm3d
        act = CReLU if activation == 'relu' else lambda: CLeakyReLU(negative_slope=0.1)
        self.block = Sequential(
            CConv2plus1d(in_channels, mid_channels, kernel_size, bias),
            norm3d(mid_channels),
            act(),
            CConv2plus1d(mid_channels, out_channels, kernel_size, bias),
            norm3d(out_channels),
            act(),
            CSEBlock3d(out_channels))
        self.shortcut = Identity() if in_channels == out_channels \
            else CConv3d(in_channels, out_channels, kernel_size=1,
                         padding=0, bias=bias)

    def forward(self, input):
        return self.block(input) + self.shortcut(input)


class _CDown2p1dBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', down='avg'):
        super().__init__()
        if down not in ('avg', 'max'):
            raise ValueError('down should be avg or max.')
        self.down_sample = \
            CAvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) if down == 'avg' \
            else CMaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.down_conv = _CDoubleConv2p1dBlock(
            in_channels, out_channels, None,
            kernel_size, bias, normalization, activation)

    def forward(self, input):
        return self.down_conv(self.down_sample(input))


class _CUp2p1dBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False,
                 normalization='instance', activation='relu', up='bilinear'):
        super().__init__()
        if up not in ('bilinear', 'convtran'):
            raise ValueError('up should be bilinear or convtran.')
        if up == 'bilinear':
            self.up_sample = CUpsamplingTrilinear3d(scale_factor=(1, 2, 2))
            self.up_conv = _CDoubleConv2p1dBlock(
                in_channels, out_channels, in_channels // 2,
                kernel_size, bias, normalization, activation)
        else:
            self.up_sample = CConvTranspose3d(
                in_channels, in_channels // 2,
                kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=bias)
            self.up_conv = _CDoubleConv2p1dBlock(
                in_channels, out_channels, None,
                kernel_size, bias, normalization, activation)

    def forward(self, input, skip_connection):
        input = self.up_sample(input)
        difft = skip_connection.shape[-3] - input.shape[-3]
        diffh = skip_connection.shape[-2] - input.shape[-2]
        diffw = skip_connection.shape[-1] - input.shape[-1]
        input = F.pad(input, (diffw // 2, diffw - diffw // 2,
                              diffh // 2, diffh - diffh // 2,
                              difft // 2, difft - difft // 2))
        return self.up_conv(torch.cat((skip_connection, input), dim=1))


class CUNet2p1d(Module):
    """UNet with (2+1)D complex convolutions. Expects (B, C, T, H, W) input."""
    def __init__(self, in_channels, out_channels, depth=4, num_filters=32,
                 kernel_size=3, bias=False, normalization='instance',
                 activation='relu', down='avg', up='bilinear'):
        super().__init__()
        self.residual = False   # see make_residual()
        self.film = None        # see make_conditional()
        self.cond = None
        self.grad_checkpoint = False   # see set_grad_checkpoint()
        factor = 2 if up == 'bilinear' else 1
        self.first = _CDoubleConv2p1dBlock(
            in_channels, num_filters, None,
            kernel_size, bias, normalization, activation)
        self.downs = ModuleList([_CDown2p1dBlock(
            num_filters, num_filters * 2, kernel_size,
            bias, normalization, activation, down)])
        num_filters *= 2
        for _ in range(1, depth - 1):
            self.downs.append(_CDown2p1dBlock(
                num_filters, num_filters * 2, kernel_size,
                bias, normalization, activation, down))
            num_filters *= 2
        self.downs.append(_CDown2p1dBlock(
            num_filters, num_filters * 2 // factor, kernel_size,
            bias, normalization, activation, down))
        self.ups = ModuleList()
        for _ in range(depth - 1):
            self.ups.append(_CUp2p1dBlock(
                num_filters * 2, num_filters // factor, kernel_size,
                bias, normalization, activation, up))
            num_filters //= 2
        self.ups.append(_CUp2p1dBlock(
            num_filters * 2, num_filters, kernel_size,
            bias, normalization, activation, up))
        self.last = CConv3d(num_filters, out_channels, kernel_size=1,
                            padding=0, bias=bias)

    def forward(self, input):
        if self.grad_checkpoint and torch.is_grad_enabled():
            output = checkpoint(self._body, input, use_reentrant=False)
        else:
            output = self._body(input)
        return _residual_input(input, output) + output if self.residual \
            else output

    def _body(self, input):
        output = self.first(input)
        if self.film is not None and self.cond is not None:
            output = output * _film_scale(self.film, self.cond, output)
        skip_connections = [output]
        for down in self.downs[:-1]:
            output = down(output)
            skip_connections.append(output)
        output = self.downs[-1](output)
        for up in self.ups:
            output = up(output, skip_connections.pop())
        return self.last(output)


def make_residual(net):
    """Turn every (C)UNet in net into input + UNet(input).

    The last layer is zero-initialised, so each UNet starts as the identity
    and the whole cascade starts at the zero-filled + data-consistency
    reconstruction, learning corrections from there. Without this, every
    UNet initially replaces its input with arbitrary output (its first layer
    normalises the input away), and the model starts below zero-filled.
    """
    for m in net.modules():
        if isinstance(m, (UNet, CUNet, CUNet2p1d, RUNet2p1d)):
            m.residual = True
            for p in m.last.parameters():
                torch.nn.init.zeros_(p)
    return net


def _unets(net):
    return [m for m in net.modules() if isinstance(m, (UNet, CUNet, CUNet2p1d, RUNet2p1d))]


def _film_scale(film, cond, x):
    """Per-sample, per-channel scale 1 + film(cond), broadcast over x."""
    g = film(cond)                                         # (B, F)
    if g.shape[0] != x.shape[0]:                           # coils in batch dim
        g = g.repeat_interleave(x.shape[0] // g.shape[0], dim=0)
    return 1 + g.view(*g.shape, *([1] * (x.dim() - 2)))


def make_conditional(net):
    """Condition every (C)UNet on the acceleration factor (FiLM).

    The acceleration is read from the sampling mask (the model's 2nd input),
    so no other interface changes: R_eff = 1 / (fraction of sampled lines),
    fed as log(R_eff). Each UNet scales the features after its first block by
    1 + Linear(log R_eff); zero-initialised, so the model starts unchanged.
    """
    for m in _unets(net):
        last = m.last if isinstance(m.last, Conv2d) else m.last.conv_r
        m.film = Linear(1, last.in_channels)
        torch.nn.init.zeros_(m.film.weight)
        torch.nn.init.zeros_(m.film.bias)

    def set_cond(module, args):
        mask = args[1]
        frac = mask.reshape(mask.shape[0], -1).float().mean(dim=1)
        cond = torch.log(1.0 / frac.clamp_min(1e-3)).unsqueeze(1)
        for m in _unets(module):
            m.cond = cond
    net.register_forward_pre_hook(set_cond)
    return net


def set_grad_checkpoint(net):
    """Recompute UNet activations in the backward pass to save GPU memory
    (needed for deep cascades); ~30% slower per step."""
    for m in _unets(net):
        m.grad_checkpoint = True
    return net


def _residual_input(input, output):
    """The part of input a residual UNet adds to its output. With more input
    than output channels (2.5D: neighbouring frames stacked frame-major as
    extra channels) this is the centre block."""
    n_out = output.shape[1]
    if input.shape[1] == n_out:
        return input
    k = input.shape[1] // n_out
    return input.narrow(1, n_out * (k // 2), n_out)


def set_temporal_conv(net, kernel=None, padding_mode=None):
    """Modify the temporal convolutions of (2+1)D UNets (the only layers that
    mix frames). kernel=1 makes the network blind to time (diagnostic);
    padding_mode='circular' wraps the cine sequence (it is periodic)."""
    for m in net.modules():
        if type(m).__name__ == 'RConv2plus1d':
            if kernel is not None and kernel != m.temporal.kernel_size[0]:
                raise ValueError('temporal kernel change not supported for '
                                 'the real UNet')
            if padding_mode is not None:
                m.temporal.padding_mode = padding_mode
            continue
        if not isinstance(m, CConv2plus1d):
            continue
        conv = m.temporal.conv_r
        if kernel is not None and kernel != conv.kernel_size[0]:
            m.temporal = CConv3d(conv.in_channels, conv.out_channels,
                                 kernel_size=(kernel, 1, 1),
                                 padding=(kernel // 2, 0, 0),
                                 bias=conv.bias is not None)
        if padding_mode is not None:
            for c in (m.temporal.conv_r, m.temporal.conv_i):
                c.padding_mode = padding_mode
    return net


# ── (2+1)D real-valued UNet (ablation of the complex image network) ──────────
# Same structure as CUNet2p1d with real layers: Conv3d / InstanceNorm3d / ReLU,
# real squeeze-excitation. Used by STC3Net(inet_real=True) on the real and
# imaginary parts as 2 channels (see RealUNet2p1dWrapper in stc3.py).

class RConv2plus1d(Module):
    """Factored (2+1)D real conv: spatial (1,k,k) -> norm -> ReLU -> temporal (k,1,1)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        self.spatial = Conv3d(in_channels, out_channels,
                              kernel_size=(1, kernel_size, kernel_size),
                              padding=(0, kernel_size // 2, kernel_size // 2),
                              bias=bias)
        self.norm = InstanceNorm3d(out_channels, affine=True,
                                   track_running_stats=True)
        self.act = ReLU()
        self.temporal = Conv3d(out_channels, out_channels,
                               kernel_size=(kernel_size, 1, 1),
                               padding=(kernel_size // 2, 0, 0), bias=bias)

    def forward(self, input):
        return self.temporal(self.act(self.norm(self.spatial(input))))


class RSEBlock3d(Module):
    """Real channel attention (pooled |x|, as CSEBlock3d)."""
    def __init__(self, channels, r=16):
        super().__init__()
        mid = max(channels // r, 4)
        self.pool = AdaptiveAvgPool3d(1)
        self.fc = Sequential(Linear(channels, mid), ReLU(),
                             Linear(mid, channels), Sigmoid())

    def forward(self, x):
        scale = self.fc(self.pool(x.abs()).flatten(1)).view(-1, x.shape[1], 1, 1, 1)
        return x * scale


class _RDoubleConv2p1dBlock(Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, bias=False):
        super().__init__()
        mid_channels = mid_channels or out_channels
        norm = lambda c: InstanceNorm3d(c, affine=True, track_running_stats=True)
        self.block = Sequential(
            RConv2plus1d(in_channels, mid_channels, kernel_size, bias),
            norm(mid_channels), ReLU(),
            RConv2plus1d(mid_channels, out_channels, kernel_size, bias),
            norm(out_channels), ReLU(),
            RSEBlock3d(out_channels))
        self.shortcut = Identity() if in_channels == out_channels \
            else Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, input):
        return self.block(input) + self.shortcut(input)


class RUNet2p1d(Module):
    """Real (2+1)D UNet mirroring CUNet2p1d (avg-pool down, bilinear up,
    spatial-only resampling). Input/output (B, C, T, H, W), real."""
    def __init__(self, in_channels, out_channels, depth=3, num_filters=23,
                 kernel_size=3, bias=False):
        super().__init__()
        self.residual = False
        self.film = self.cond = None
        self.grad_checkpoint = False
        self.first = _RDoubleConv2p1dBlock(in_channels, num_filters, None,
                                           kernel_size, bias)
        pool = lambda: AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.downs = ModuleList()
        f = num_filters
        for d in range(depth):
            out = f * 2 if d < depth - 1 else f      # bilinear: last down keeps f
            self.downs.append(Sequential(pool(), _RDoubleConv2p1dBlock(
                f, out, None, kernel_size, bias)))
            f = out if d < depth - 1 else f
        self.ups = ModuleList()
        for d in range(depth):
            out = f // 2 if d < depth - 1 else f
            self.ups.append(_RDoubleConv2p1dBlock(f * 2, out, f, kernel_size, bias))
            f = out
        self.up = Upsample(scale_factor=(1, 2, 2), mode='trilinear',
                           align_corners=True)
        self.last = Conv3d(f, out_channels, kernel_size=1, bias=bias)

    def forward(self, input):
        if self.grad_checkpoint and torch.is_grad_enabled():
            output = checkpoint(self._body, input, use_reentrant=False)
        else:
            output = self._body(input)
        return input + output if self.residual else output

    def _body(self, x):
        x = self.first(x)
        skips = [x]
        for down in self.downs[:-1]:
            x = down(x)
            skips.append(x)
        x = self.downs[-1](x)
        for up in self.ups:
            s = skips.pop()
            x = self.up(x)
            dt, dh, dw = (s.shape[-3] - x.shape[-3], s.shape[-2] - x.shape[-2],
                          s.shape[-1] - x.shape[-1])
            x = F.pad(x, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2,
                          dt // 2, dt - dt // 2))
            x = up(torch.cat((s, x), dim=1))
        return self.last(x)
