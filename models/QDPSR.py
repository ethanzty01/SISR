import torch
import torch.nn as nn
import torch.nn.functional as F

from .DPSR import bilinear_skip


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def _positive_ste(value, eps):
    """Use a positive value in forward while preserving the LSQ gradient."""
    projected = value.clamp_min(eps)
    return value + (projected - value).detach()

class LSQPlusActQuant(nn.Module):
    def __init__(self, bitwidth=4, channels=1, signed=True, eps=1e-8):
        super().__init__()
        self.bitwidth = int(bitwidth)
        self.signed = bool(signed)
        self.eps = float(eps)
        self.disabled = self.bitwidth == -1 or self.bitwidth >= 32
        self.channels = int(channels)
        self.s = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        self.initialized = False

        if self.signed:
            self.qn = -float(1 << (self.bitwidth - 1))
            self.qp = float((1 << (self.bitwidth - 1)) - 1)
        else:
            self.qn = 0.0
            self.qp = float((1 << self.bitwidth) - 1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        s_key = prefix + 's'
        raw_s_key = prefix + 'raw_s'
        if s_key not in state_dict and raw_s_key in state_dict:
            raw_s = state_dict.pop(raw_s_key)
            state_dict[s_key] = F.softplus(raw_s) + self.eps
        # Grouped checkpoints store a persistent channel->group map; expanding
        # per-group values through it preserves the quantized function exactly.
        group_index = state_dict.pop(prefix + 'group_index', None)
        for key, target in ((s_key, self.s), (prefix + 'beta', self.beta)):
            if key not in state_dict or state_dict[key].shape == target.shape:
                continue
            flat = state_dict[key].detach().reshape(-1)
            if group_index is not None and flat.numel() == int(group_index.max()) + 1:
                flat = flat[group_index.to(flat.device).long()]
            elif flat.numel() > 0 and self.channels % flat.numel() == 0:
                flat = flat.repeat_interleave(self.channels // flat.numel())
            # Unsupported counts retain a shape mismatch for strict loading.
            if flat.numel() == self.channels:
                state_dict[key] = flat.reshape_as(target).clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.initialized = True

    def _init_from_tensor(self, x):
        if self.disabled:
            self.initialized = True
            return
        x = x.float()
        if x.dim() == 2:
            channel_values = x.detach()
        elif x.dim() >= 2:
            channel_values = x.detach().movedim(1, 0).reshape(x.shape[1], -1)
        else:
            raise ValueError('LSQ+ activation calibration requires a channel dimension')

        active_channels = channel_values.shape[0]
        if active_channels > self.channels:
            raise ValueError(f'calibration has {active_channels} channels, expected at most {self.channels}')
        tails = channel_values.new_tensor([0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05])
        for channel in range(active_channels):
            values = channel_values[channel]
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                raise ValueError(f'cannot initialize LSQ+ channel {channel} from non-finite values')
            if values.numel() > 8192:
                indices = torch.linspace(0, values.numel() - 1, 8192, device=values.device).long()
                values = values[indices]

            lower_candidates = torch.quantile(values, tails)
            upper_candidates = torch.quantile(values, 1.0 - tails)
            lower = lower_candidates[:, None].expand(-1, tails.numel()).reshape(-1)
            upper = upper_candidates[None, :].expand(tails.numel(), -1).reshape(-1)
            step = ((upper - lower) / (self.qp - self.qn)).clamp_min(self.eps)
            beta = lower - self.qn * step
            normalized = (values.unsqueeze(0) - beta.unsqueeze(1)) / step.unsqueeze(1)
            reconstructed = normalized.round().clamp(self.qn, self.qp) * step.unsqueeze(1) + beta.unsqueeze(1)
            best_index = (reconstructed - values.unsqueeze(0)).square().mean(dim=1).argmin()
            self.s.data[0, channel, 0, 0].copy_(step[best_index])
            self.beta.data[0, channel, 0, 0].copy_(beta[best_index])
        self.initialized = True

    def initialize_from_samples(self, values):
        self._init_from_tensor(values)

    def forward(self, x, active_channels=None):
        if self.disabled:
            return x
        # Quantization arithmetic stays FP32 even when convolutions use AMP.
        x = x.float()
        if not self.initialized:
            self._init_from_tensor(x)
        c = x.shape[1] if active_channels is None else int(active_channels)
        if c != x.shape[1] or not 0 < c <= self.channels:
            raise ValueError(f'invalid active channel count: {c}')
        n = x.numel() / c
        g = 1.0 / ((n * self.qp) ** 0.5)
        s = ScaleGrad.apply(_positive_ste(self.s[:, :c], self.eps), g)
        beta = ScaleGrad.apply(self.beta[:, :c], g)
        x_hat = (x - beta) / s
        x_q = torch.clamp(RoundSTE.apply(x_hat), self.qn, self.qp)
        return x_q * s + beta

    @torch.no_grad()
    def project_scale_(self):
        self.s.clamp_(min=self.eps)

class LSQPlusWeightQuant(nn.Module):
    """Signed affine weight quantization with one scale/offset per output channel."""
    def __init__(self, bitwidth=4, out_channels=1, eps=1e-8):
        super().__init__()
        self.bitwidth = int(bitwidth)
        self.eps = float(eps)
        self.disabled = self.bitwidth == -1 or self.bitwidth >= 32
        self.qn = -float(1 << (self.bitwidth - 1))
        self.qp = float((1 << (self.bitwidth - 1)) - 1)
        self.s = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(out_channels, 1, 1, 1))
        self.initialized = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        s_key = prefix + 's'
        raw_s_key = prefix + 'raw_s'
        if s_key not in state_dict and raw_s_key in state_dict:
            state_dict[s_key] = F.softplus(state_dict.pop(raw_s_key)) + self.eps
        # Symmetric checkpoints predate beta; a zero offset is equivalent.
        if prefix + 'beta' not in state_dict:
            state_dict[prefix + 'beta'] = torch.zeros_like(self.beta)
        # Expand legacy per-group parameters to per-channel form.
        for key, target in ((s_key, self.s), (prefix + 'beta', self.beta)):
            if key not in state_dict or state_dict[key].shape == target.shape:
                continue
            flat = state_dict[key].detach().reshape(-1)
            if flat.numel() > 0 and target.numel() % flat.numel() == 0:
                flat = flat.repeat_interleave(target.numel() // flat.numel())
            if flat.numel() == target.numel():
                state_dict[key] = flat.reshape_as(target).clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.initialized = True

    def _init_from_tensor(self, w):
        if self.disabled:
            self.initialized = True
            return
        reduce_dims = tuple(range(1, w.dim()))
        w_mean = w.detach().mean(dim=reduce_dims, keepdim=True)
        w_std = w.detach().std(dim=reduce_dims, keepdim=True).clamp(min=self.eps)
        self.s.data.copy_((2.0 * w_std) / (self.qp ** 0.5))
        self.beta.data.copy_(w_mean)
        self.initialized = True

    def forward(self, w, active_out_channels=None):
        if self.disabled:
            return w
        if not self.initialized:
            self._init_from_tensor(w)
        if active_out_channels is None:
            active_out_channels = w.shape[0]
        c = int(active_out_channels)
        s = self.s[:c]
        beta = self.beta[:c]
        n = w[0].numel()
        g = 1.0 / ((n * self.qp) ** 0.5)
        s = ScaleGrad.apply(_positive_ste(s, self.eps), g)
        beta = ScaleGrad.apply(beta, g)
        w_hat = (w - beta) / s
        w_q = torch.clamp(RoundSTE.apply(w_hat), self.qn, self.qp)
        return w_q * s + beta

    @torch.no_grad()
    def project_scale_(self):
        self.s.clamp_(min=self.eps)

class QConv2dLSQP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
        weight_bitwidth=4,
        activation_bitwidth=4,
        act_signed=True,
        quantize_input=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.act_quant = LSQPlusActQuant(
            bitwidth=activation_bitwidth,
            channels=in_channels,
            signed=act_signed,
        )
        self.weight_quant = LSQPlusWeightQuant(
            bitwidth=weight_bitwidth,
            out_channels=out_channels,
        )
        self.quantize_input = bool(quantize_input)
        self.quantization_enabled = True

    def set_quantization_enabled(self, enabled: bool):
        self.quantization_enabled = bool(enabled)

    def _maybe_quantize_input(self, x):
        if self.quantize_input:
            return self.act_quant(x, active_channels=x.shape[1])
        return x

    def forward(self, x, active_out_channels=None):
        active_in_channels = x.shape[1]
        active_out_channels = self.conv.out_channels if active_out_channels is None else int(active_out_channels)
        if not 0 < active_in_channels <= self.conv.in_channels:
            raise ValueError(f'invalid active input channels: {active_in_channels}')
        if not 0 < active_out_channels <= self.conv.out_channels:
            raise ValueError(f'invalid active output channels: {active_out_channels}')

        if self.conv.groups == 1:
            weight = self.conv.weight[:active_out_channels, :active_in_channels]
            groups = 1
        else:
            if self.conv.groups != self.conv.in_channels or active_in_channels != active_out_channels:
                raise ValueError('only matching-width depthwise channel slicing is supported')
            weight = self.conv.weight[:active_out_channels]
            groups = active_in_channels

        if self.quantization_enabled:
            qx = self._maybe_quantize_input(x)
            if not self.weight_quant.initialized:
                self.weight_quant._init_from_tensor(self.conv.weight)
            qw = self.weight_quant(weight, active_out_channels=active_out_channels)
        else:
            qx, qw = x, weight
        bias = self.conv.bias[:active_out_channels] if self.conv.bias is not None else None
        return F.conv2d(
            qx,
            qw,
            bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=groups,
        )

class QConvTranspose2dLSQP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=False,
        weight_bitwidth=4,
        activation_bitwidth=4,
        act_signed=True,
        quantize_input=True,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        self.act_quant = LSQPlusActQuant(
            bitwidth=activation_bitwidth,
            channels=in_channels,
            signed=act_signed,
        )
        self.weight_quant = LSQPlusWeightQuant(
            bitwidth=weight_bitwidth,
            out_channels=in_channels,
        )
        self.quantize_input = bool(quantize_input)
        self.quantization_enabled = True

    def set_quantization_enabled(self, enabled: bool):
        self.quantization_enabled = bool(enabled)

    def _maybe_quantize_input(self, x):
        if self.quantize_input:
            return self.act_quant(x, active_channels=x.shape[1])
        return x

    def forward(self, x, active_out_channels=None):
        if active_out_channels is not None and int(active_out_channels) != self.conv.out_channels:
            raise ValueError('QConvTranspose2dLSQP does not support output channel slicing')
        if self.quantization_enabled:
            qx = self._maybe_quantize_input(x)
            if not self.weight_quant.initialized:
                self.weight_quant._init_from_tensor(self.conv.weight)
            qw = self.weight_quant(self.conv.weight)
        else:
            qx, qw = x, self.conv.weight
        return F.conv_transpose2d(
            qx,
            qw,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            output_padding=self.conv.output_padding,
        )

class QUSPReLU(nn.PReLU):
    """PReLU that shares leading parameters with a quantized subnet path."""

    def forward(self, x):
        return F.prelu(x, self.weight[: x.shape[1]])

class QBlock(nn.Module):
    def __init__(
        self,
        fea_dim,
        bias=False,
        weight_bitwidth=4,
        activation_bitwidth=4,
    ):
        super().__init__()
        kwargs = dict(
            kernel_size=1,
            padding=0,
            bias=bias,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=activation_bitwidth,
        )
        self.projection1 = QConv2dLSQP(fea_dim, fea_dim, **kwargs)
        self.projection2 = QConv2dLSQP(fea_dim, fea_dim, **kwargs)
        depthwise_kwargs = dict(
            kernel_size=3,
            padding=1,
            bias=bias,
            groups=fea_dim,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=activation_bitwidth,
        )
        self.filter1 = QConv2dLSQP(fea_dim, fea_dim, **depthwise_kwargs)
        self.filter2 = QConv2dLSQP(fea_dim, fea_dim, **depthwise_kwargs)
        self.act1 = QUSPReLU(num_parameters=fea_dim, init=0.25)
        self.act2 = QUSPReLU(num_parameters=fea_dim, init=0.25)

    def forward(self, x):
        channels = x.shape[1]
        y = self.filter1(x, active_out_channels=channels)
        y = self.projection1(y, active_out_channels=channels)
        y = self.act1(y)
        y = self.filter2(y, active_out_channels=channels)
        y = self.projection2(y, active_out_channels=channels)
        y = self.act2(y)
        return y

class QDPSR(nn.Module):
    def __init__(
        self,
        scale,
        in_dim,
        fea_dim,
        num_blocks=5,
        bias=False,
        weight_bitwidth=4,
        activation_bitwidth=4,
        subnet_channels=16,
        use_bilinear=True,
    ):
        super().__init__()

        self.use_bilinear = bool(use_bilinear)
        self.scale = scale
        self.fea_dim = int(fea_dim)
        self.subnet_channels = int(subnet_channels)
        self.num_blocks = int(num_blocks)
        if not 0 < self.subnet_channels < self.fea_dim:
            raise ValueError(f'subnet_channels must be in [1, {self.fea_dim - 1}]')

        self.head = QConv2dLSQP(
            in_channels=in_dim,
            out_channels=fea_dim,
            kernel_size=1,
            padding=0,
            bias=bias,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=8,
            quantize_input=False,
        )
        # RGB/255 already represents 8-bit input; keep legacy quantizer keys inactive.
        self.head.act_quant.requires_grad_(False)

        self.body = nn.ModuleList(
            QBlock(
                fea_dim,
                bias=bias,
                weight_bitwidth=weight_bitwidth,
                activation_bitwidth=activation_bitwidth,
            )
            for _ in range(num_blocks)
        )

        self.tail = QConv2dLSQP(
            in_channels=fea_dim,
            out_channels=in_dim * scale ** 2,
            kernel_size=1,
            padding=0,
            bias=bias,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=activation_bitwidth,
        )

        self.upsampler = nn.PixelShuffle(scale)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, a=0.25)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.tail.conv.weight)
        if self.tail.conv.bias is not None:
            nn.init.zeros_(self.tail.conv.bias)

    @property
    def supported_channels(self):
        return (self.fea_dim, self.subnet_channels)

    def forward(self, x, channels=None, use_bilinear=None):
        channels = self.fea_dim if channels is None else int(channels)
        if channels not in self.supported_channels:
            raise ValueError(f'channels must be one of {self.supported_channels}, got {channels}')
        # Feed raw RGB/grayscale normalized by 255 directly into the head.
        y = self.head(x, active_out_channels=channels)
        for block in self.body:
            y = block(y)
        y = self.upsampler(self.tail(y))
        enabled = self.use_bilinear if use_bilinear is None else use_bilinear
        return y + bilinear_skip(x, self.scale) if enabled else y

    def set_quantization_enabled(self, enabled):
        for module in self.modules():
            if isinstance(module, QConv2dLSQP):
                module.set_quantization_enabled(enabled)

    @torch.no_grad()
    def project_quantization_parameters(self):
        for module in self.modules():
            if isinstance(module, QConv2dLSQP):
                module.act_quant.project_scale_()
                module.weight_quant.project_scale_()

    def param_num(self):
        return sum(parameter.numel() for parameter in self.parameters())


class QFSRCNN(nn.Module):
    def __init__(
        self,
        scale_factor,
        num_channels=1,
        d=56,
        s=12,
        m=4,
        weight_bitwidth=4,
        activation_bitwidth=4,
    ):
        super().__init__()
        self.scale = scale_factor
        self.fea_dim = d
        self.first_part = nn.Sequential(
            QConv2dLSQP(
                num_channels,
                d,
                kernel_size=5,
                padding=5 // 2,
                bias=True,
                weight_bitwidth=weight_bitwidth,
                activation_bitwidth=activation_bitwidth,
            ),
            nn.PReLU(d),
        )
        self.mid_part = [QConv2dLSQP(d, s, kernel_size=1, bias=True, weight_bitwidth=weight_bitwidth, activation_bitwidth=activation_bitwidth), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([
                QConv2dLSQP(s, s, kernel_size=3, padding=3 // 2, bias=True, weight_bitwidth=weight_bitwidth, activation_bitwidth=activation_bitwidth),
                nn.PReLU(s),
            ])
        self.mid_part.extend([
            QConv2dLSQP(s, d, kernel_size=1, bias=True, weight_bitwidth=weight_bitwidth, activation_bitwidth=activation_bitwidth),
            nn.PReLU(d),
        ])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = QConvTranspose2dLSQP(
            d,
            num_channels,
            kernel_size=9,
            stride=scale_factor,
            padding=9 // 2,
            output_padding=scale_factor - 1,
            bias=True,
            weight_bitwidth=weight_bitwidth,
            activation_bitwidth=activation_bitwidth,
        )

    def forward(self, x, channels=None):
        x = self.first_part(x)
        x = self.mid_part(x)
        return self.last_part(x)

    def set_quantization_enabled(self, enabled):
        for module in self.modules():
            if isinstance(module, (QConv2dLSQP, QConvTranspose2dLSQP)):
                module.set_quantization_enabled(enabled)

    @torch.no_grad()
    def project_quantization_parameters(self):
        for module in self.modules():
            if isinstance(module, (QConv2dLSQP, QConvTranspose2dLSQP)):
                module.act_quant.project_scale_()
                module.weight_quant.project_scale_()

    def param_num(self):
        return sum(parameter.numel() for parameter in self.parameters())


def build_qdpsr(
    scale,
    in_dim,
    fea_dim,
    num_blocks=5,
    bias=False,
    weight_bitwidth=4,
    activation_bitwidth=4,
    subnet_channels=16,
    use_bilinear=True,
):
    return QDPSR(
        scale=scale,
        in_dim=in_dim,
        fea_dim=fea_dim,
        num_blocks=num_blocks,
        bias=bias,
        weight_bitwidth=weight_bitwidth,
        activation_bitwidth=activation_bitwidth,
        subnet_channels=subnet_channels,
        use_bilinear=use_bilinear,
    )


def build_qfsrcnn(
    scale,
    in_dim,
    weight_bitwidth=4,
    activation_bitwidth=4,
):
    return QFSRCNN(
        scale_factor=scale,
        num_channels=in_dim,
        weight_bitwidth=weight_bitwidth,
        activation_bitwidth=activation_bitwidth,
    )
