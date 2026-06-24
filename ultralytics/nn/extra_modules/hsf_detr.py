"""HSF-DETR custom modules: ETB, HCFM and GCConv only.

This file intentionally contains only the three modules used in the paper:
ETB, HCFM and GCConv. Other experimental modules from the previous fork were
removed to make the implementation easier to read and reproduce.
"""

from typing import Optional, Tuple, Union
import numbers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, RepConv


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def _to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, hw, c = x.shape
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class _BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class _WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class _LayerNorm2d(nn.Module):
    def __init__(self, dim, layer_norm_type='BiasFree'):
        super().__init__()
        self.body = _BiasFreeLayerNorm(dim) if layer_norm_type == 'BiasFree' else _WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    b, c, h, w = x.shape
    assert c % heads == 0, f'channels {c} must be divisible by heads {heads}'
    return x.reshape(b, heads, c // heads, h * w)


def _merge_heads(x: torch.Tensor, heads: int, h: int, w: int) -> torch.Tensor:
    b, _, c, _ = x.shape
    return x.reshape(b, heads * c, h, w)


def _complex_softmax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    real = F.softmax(input_tensor.real, dim=dim)
    imag = F.softmax(input_tensor.imag, dim=dim)
    return torch.complex(real, imag)


class _ETBFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=4, bias=False):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim * 4, dim, kernel_size=1, bias=bias)
        hidden = max(dim // 16, 1)
        self.weight = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.BatchNorm2d(hidden),
            nn.ReLU(True),
            nn.Conv2d(hidden, dim, 1, bias=True),
            nn.Sigmoid(),
        )
        self.weight1 = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, bias=True),
            nn.BatchNorm2d(hidden),
            nn.ReLU(True),
            nn.Conv2d(hidden, dim * 2, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        xf_fft = torch.fft.fft2(x.float())
        x_f = torch.abs(self.weight(xf_fft.real) * xf_fft)
        x_f_gelu = F.gelu(x_f) * x_f
        x_s = self.dwconv1(x)
        x_s_gelu = F.gelu(x_s) * x_s
        x_mix = torch.fft.fft2(torch.cat((x_f_gelu, x_s_gelu), dim=1))
        x_f = torch.abs(torch.fft.ifft2(self.weight1(x_mix.real) * x_mix))
        x_s = self.dwconv2(torch.cat((x_f_gelu, x_s_gelu), dim=1))
        return self.project_out(torch.cat((x_f, x_s), dim=1))


class _FrequencySelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        hidden = max(dim // 16, 1)
        self.weight = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.BatchNorm2d(hidden),
            nn.ReLU(True),
            nn.Conv2d(hidden, dim, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_fft = torch.fft.fft2(x.float())
        q_f = _split_heads(x_fft, self.num_heads)
        k_f = _split_heads(x_fft, self.num_heads)
        v_f = _split_heads(x_fft, self.num_heads)
        q_f = F.normalize(q_f, dim=-1)
        k_f = F.normalize(k_f, dim=-1)
        attn = (q_f @ k_f.transpose(-2, -1)) * self.temperature
        attn = _complex_softmax(attn, dim=-1)
        out_f = torch.abs(torch.fft.ifft2(attn @ v_f))
        out_f = _merge_heads(out_f, self.num_heads, h, w)
        out_f_l = torch.abs(torch.fft.ifft2(self.weight(x_fft.real) * x_fft))
        return self.project_out(torch.cat((out_f, out_f_l), dim=1))


class _SpatialSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.k1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.v1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.q3 = nn.Conv2d(dim, dim // 2, 3, 1, 1, groups=max(dim // 2, 1), bias=bias)
        self.k3 = nn.Conv2d(dim, dim // 2, 3, 1, 1, groups=max(dim // 2, 1), bias=bias)
        self.v3 = nn.Conv2d(dim, dim // 2, 3, 1, 1, groups=max(dim // 2, 1), bias=bias)
        self.q5 = nn.Conv2d(dim, dim // 2, 5, 1, 2, groups=max(dim // 2, 1), bias=bias)
        self.k5 = nn.Conv2d(dim, dim // 2, 5, 1, 2, groups=max(dim // 2, 1), bias=bias)
        self.v5 = nn.Conv2d(dim, dim // 2, 5, 1, 2, groups=max(dim // 2, 1), bias=bias)
        self.conv3 = nn.Conv2d(dim, dim // 2, 3, 1, 1, groups=max(dim // 2, 1), bias=bias)
        self.conv5 = nn.Conv2d(dim, dim // 2, 5, 1, 2, groups=max(dim // 2, 1), bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q = torch.cat((self.q3(self.q1(x)), self.q5(self.q1(x))), dim=1)
        k = torch.cat((self.k3(self.k1(x)), self.k5(self.k1(x))), dim=1)
        v = torch.cat((self.v3(self.v1(x)), self.v5(self.v1(x))), dim=1)
        q = _split_heads(q, self.num_heads)
        k = _split_heads(k, self.num_heads)
        v = _split_heads(v, self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = _merge_heads(attn @ v, self.num_heads, h, w)
        local = torch.cat((self.conv3(x), self.conv5(x)), dim=1)
        return self.project_out(torch.cat((out, local), dim=1))


class ETB(nn.Module):
    """Entanglement Transformer Block used to replace AIFI in RT-DETR."""

    def __init__(self, dim=128, num_heads=4, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = _LayerNorm2d(dim, LayerNorm_type)
        self.attn_s = _SpatialSelfAttention(dim, num_heads, bias)
        self.attn_f = _FrequencySelfAttention(dim, num_heads, bias)
        self.norm2 = _LayerNorm2d(dim, LayerNorm_type)
        self.ffn = _ETBFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        nx = self.norm1(x)
        x = x + self.attn_f(nx) + self.attn_s(nx)
        x = x + self.ffn(self.norm2(x))
        return x


class LocalGlobalAttention(nn.Module):
    """Patch-level local-global attention branch used inside HCFM."""

    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = nn.Parameter(torch.randn(output_dim))
        self.top_down_transform = nn.Parameter(torch.eye(output_dim))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        b, h, w, c = x.shape
        p = self.patch_size
        # Pad only when needed, so the module remains robust to non-divisible shapes.
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = x.shape[1], x.shape[2]
        patches = x.unfold(1, p, p).unfold(2, p, p).reshape(b, -1, p * p, c)
        patches = patches.mean(dim=-1)
        local = self.mlp2(self.norm(self.mlp1(patches)))
        attn = F.softmax(local, dim=-1)
        local = local * attn
        cos_sim = F.normalize(local, dim=-1) @ F.normalize(self.prompt[None, :, None], dim=1)
        local = local * cos_sim.clamp(0, 1)
        local = local @ self.top_down_transform
        local = local.reshape(b, hp // p, wp // p, self.output_dim).permute(0, 3, 1, 2)
        local = F.interpolate(local, size=(hp, wp), mode='bilinear', align_corners=False)
        local = local[:, :, :h, :w]
        return self.conv(local)


class HCFM(nn.Module):
    """Hierarchical Context Fusion Module.

    It replaces simple Concat operations in RT-DETR neck fusion.
    """

    def __init__(self, inc, ouc, group=False):
        super().__init__()
        ch1, ch2 = inc
        hidc = ouc // 2
        self.lgb1_local = LocalGlobalAttention(hidc, 2)
        self.lgb1_global = LocalGlobalAttention(hidc, 4)
        self.lgb2_local = LocalGlobalAttention(hidc, 2)
        self.lgb2_global = LocalGlobalAttention(hidc, 4)
        self.W_x1 = Conv(ch1, hidc, 1, act=False)
        self.W_x2 = Conv(ch2, hidc, 1, act=False)
        self.W = Conv(hidc, ouc, 3, g=4)
        self.conv_squeeze = Conv(ouc * 3, ouc, 1)
        self.rep_conv = RepConv(ouc, ouc, 3, g=(16 if group else 1))
        self.conv_final = Conv(ouc, ouc, 1)

    def forward(self, inputs):
        x1, x2 = inputs
        wx1 = self.W_x1(x1)
        wx2 = self.W_x2(x2)
        bp = self.W(wx1 + wx2)
        x1 = torch.cat((self.lgb1_local(wx1), self.lgb1_global(wx1)), dim=1)
        x2 = torch.cat((self.lgb2_local(wx2), self.lgb2_global(wx2)), dim=1)
        return self.conv_final(self.rep_conv(self.conv_squeeze(torch.cat((x1, x2, bp), dim=1))))


class _Block1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        if deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, 1, stride, padding, bias=True)
        else:
            self.conv1 = Conv(in_channels, out_channels, k=1, s=stride, p=padding, act=False)
            self.conv2 = Conv(out_channels, out_channels, k=1, s=1, p=padding, act=False)

    def forward(self, x):
        return self.conv(x) if self.deploy else self.conv2(self.conv1(x))

    @staticmethod
    def _fuse_bn_tensor(conv):
        kernel = conv.conv.weight
        bias = conv.conv.bias
        rm, rv = conv.bn.running_mean, conv.bn.running_var
        gamma, beta, eps = conv.bn.weight, conv.bn.bias, conv.bn.eps
        std = (rv + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - rm) * gamma / std if bias is not None else beta - rm * gamma / std

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride, self.padding, bias=True)
        self.conv.weight.data = torch.einsum('oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1)
        self.conv.bias.data = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        del self.conv1, self.conv2
        self.deploy = True


class _Block3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        if deploy:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=True)
        else:
            self.conv1 = Conv(in_channels, out_channels, k=3, s=stride, p=padding, act=False)
            self.conv2 = Conv(out_channels, out_channels, k=1, s=1, p=0, act=False)

    def forward(self, x):
        return self.conv(x) if self.deploy else self.conv2(self.conv1(x))

    @staticmethod
    def _fuse_bn_tensor(conv):
        kernel = conv.conv.weight
        bias = conv.conv.bias
        rm, rv = conv.bn.running_mean, conv.bn.running_var
        gamma, beta, eps = conv.bn.weight, conv.bn.bias, conv.bn.eps
        std = (rv + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - rm) * gamma / std if bias is not None else beta - rm * gamma / std

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 3, self.stride, self.padding, bias=True)
        self.conv.weight.data = torch.einsum('oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1)
        self.conv.bias.data = bias2 + (bias1.view(1, -1, 1, 1) * kernel2).sum(3).sum(2).sum(1)
        del self.conv1, self.conv2
        self.deploy = True


class GCConv(nn.Module):
    """Re-parameterized GCConv used to replace stride-2 downsampling Conv."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 1,
                 padding_mode: Optional[str] = 'zeros', deploy: bool = False):
        super().__init__()
        assert kernel_size == 3, 'GCConv only supports a 3x3 kernel in this implementation.'
        assert padding == 1, 'GCConv expects padding=1 for 3x3 same-padding.'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.deploy = deploy
        self.act = nn.SiLU()
        if deploy:
            self.reparam_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True,
                                         padding_mode=padding_mode)
        else:
            self.path_residual = nn.BatchNorm2d(in_channels) if (out_channels == in_channels and stride == 1) else None
            self.path_3x3_1 = _Block3x3(in_channels, out_channels, stride=stride, padding=padding)
            self.path_3x3_2 = _Block3x3(in_channels, out_channels, stride=stride, padding=padding)
            self.path_1x1 = _Block1x1(in_channels, out_channels, stride=stride, padding=padding - kernel_size // 2)

    def forward(self, x):
        if hasattr(self, 'reparam_3x3'):
            return self.act(self.reparam_3x3(x))
        residual = 0 if self.path_residual is None else self.path_residual(x)
        return self.act(self.path_3x3_1(x) + self.path_3x3_2(x) + self.path_1x1(x) + residual)

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        return 0 if kernel1x1 is None else F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, conv):
        if conv is None:
            return 0, 0
        if isinstance(conv, Conv):
            kernel = conv.conv.weight
            rm, rv = conv.bn.running_mean, conv.bn.running_var
            gamma, beta, eps = conv.bn.weight, conv.bn.bias, conv.bn.eps
        else:
            assert isinstance(conv, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                kernel_value = np.zeros((self.in_channels, self.in_channels, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(conv.weight.device)
            kernel = self.id_tensor
            rm, rv = conv.running_mean, conv.running_var
            gamma, beta, eps = conv.weight, conv.bias, conv.eps
        std = (rv + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - rm * gamma / std

    def get_equivalent_kernel_bias(self):
        self.path_3x3_1.switch_to_deploy()
        k31, b31 = self.path_3x3_1.conv.weight.data, self.path_3x3_1.conv.bias.data
        self.path_3x3_2.switch_to_deploy()
        k32, b32 = self.path_3x3_2.conv.weight.data, self.path_3x3_2.conv.bias.data
        self.path_1x1.switch_to_deploy()
        k11, b11 = self.path_1x1.conv.weight.data, self.path_1x1.conv.bias.data
        kid, bid = self._fuse_bn_tensor(self.path_residual)
        return k31 + k32 + self._pad_1x1_to_3x3_tensor(k11) + kid, b31 + b32 + b11 + bid

    def switch_to_deploy(self):
        if hasattr(self, 'reparam_3x3'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_3x3 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=True)
        self.reparam_3x3.weight.data = kernel
        self.reparam_3x3.bias.data = bias
        del self.path_3x3_1, self.path_3x3_2, self.path_1x1
        if hasattr(self, 'path_residual'):
            del self.path_residual
        if hasattr(self, 'id_tensor'):
            del self.id_tensor
        self.deploy = True
