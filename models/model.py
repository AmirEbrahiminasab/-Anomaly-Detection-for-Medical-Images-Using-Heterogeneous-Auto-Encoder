import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import models
from typing import List, Dict
import math

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # --- FIX: Switched to ResNet-18 as used in the paper's official implementation ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1 # Output C=64
        self.layer2 = resnet.layer2 # Output C=128
        self.layer3 = resnet.layer3 # Output C=256
        self.layer4 = resnet.layer4 # Output C=512

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stage0(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return {'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4}

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class MSTB(nn.Module):
    def __init__(self, in_channels, num_heads, h, w, p1_size, p2_size):
        super().__init__()
        assert in_channels % num_heads == 0, "C % heads != 0"
        assert num_heads % 2 == 0, "num_heads must be even"
        assert h % p1_size == 0 and w % p1_size == 0
        assert h % p2_size == 0 and w % p2_size == 0

        self.num_heads = num_heads
        self.head_dim  = in_channels // num_heads
        self.scale     = self.head_dim ** -0.5
        self.h, self.w = h, w

        N  = h * w
        N1 = (h // p1_size) * (w // p1_size)
        N2 = (h // p2_size) * (w // p2_size)

        self.pos_local = nn.Parameter(torch.zeros(1, N,  in_channels))
        self.pos_p1    = nn.Parameter(torch.zeros(1, N1, in_channels))
        self.pos_p2    = nn.Parameter(torch.zeros(1, N2, in_channels))
        trunc_normal_(self.pos_local, std=.02)
        trunc_normal_(self.pos_p1,    std=.02)
        trunc_normal_(self.pos_p2,    std=.02)

        self.patch_p1 = nn.Conv2d(in_channels, in_channels, kernel_size=p1_size, stride=p1_size)
        self.patch_p2 = nn.Conv2d(in_channels, in_channels, kernel_size=p2_size, stride=p2_size)

        self.to_q     = nn.Linear(in_channels, in_channels)
        self.to_k_p1  = nn.Linear(in_channels, in_channels // 2)
        self.to_v_p1  = nn.Linear(in_channels, in_channels // 2)
        self.to_k_p2  = nn.Linear(in_channels, in_channels // 2)
        self.to_v_p2  = nn.Linear(in_channels, in_channels // 2)

        self.norm_q   = nn.LayerNorm(in_channels)
        self.norm_kv1 = nn.LayerNorm(in_channels)
        self.norm_kv2 = nn.LayerNorm(in_channels)

        self.to_out   = nn.Linear(in_channels, in_channels)
        self.norm2    = nn.LayerNorm(in_channels)
        self.mlp      = nn.Sequential(
            nn.Linear(in_channels, 4 * in_channels),
            nn.GELU(),
            nn.Linear(4 * in_channels, in_channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.h and W == self.w, "positional embeddings sized for (h,w)"

        x_flat = x.flatten(2).transpose(1, 2)
        q_in   = self.norm_q(x_flat + self.pos_local)
        q      = self.to_q(q_in).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q1, q2 = torch.chunk(q, 2, dim=1)

        t1 = self.patch_p1(x).flatten(2).transpose(1, 2) + self.pos_p1
        t1 = self.norm_kv1(t1)
        k1 = self.to_k_p1(t1).reshape(B, -1, self.num_heads // 2, self.head_dim).permute(0, 2, 1, 3)
        v1 = self.to_v_p1(t1).reshape(B, -1, self.num_heads // 2, self.head_dim).permute(0, 2, 1, 3)

        t2 = self.patch_p2(x).flatten(2).transpose(1, 2) + self.pos_p2
        t2 = self.norm_kv2(t2)
        k2 = self.to_k_p2(t2).reshape(B, -1, self.num_heads // 2, self.head_dim).permute(0, 2, 1, 3)
        v2 = self.to_v_p2(t2).reshape(B, -1, self.num_heads // 2, self.head_dim).permute(0, 2, 1, 3)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        out1  = (attn1.softmax(dim=-1) @ v1).transpose(1, 2).reshape(B, H*W, C // 2)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        out2  = (attn2.softmax(dim=-1) @ v2).transpose(1, 2).reshape(B, H*W, C // 2)

        Z = torch.cat([out1, out2], dim=-1)
        Z = self.to_out(Z)

        Z = Z + x_flat
        Z = Z + self.mlp(self.norm2(Z))

        return Z.transpose(1, 2).reshape(B, C, H, W)

class HybridModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int, h: int, w: int, p_sizes: tuple):
        super(HybridModule, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, stride=1)
        )
        self.transformer_block = MSTB(
            in_channels=in_channels // 2, num_heads=num_heads,
            h=h, w=w, p1_size=p_sizes[0], p2_size=p_sizes[1]
        )
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv, x_trans = self.conv1_1(x), self.conv1_2(x)
        f_conv, f_trans = self.conv_block(x_conv), self.transformer_block(x_trans)
        f_cat = torch.cat([f_conv, f_trans], dim=1)
        return self.upsample(self.conv_out(f_cat))

class Decoder(nn.Module):
    def __init__(self, image_size=256):
        super(Decoder, self).__init__()
        s = image_size
        s32, s16, s8 = s // 32, s // 16, s // 8

        # --- FIX: Adjusted channel sizes to match ResNet-18 ---
        self.stage1 = HybridModule(in_channels=512, out_channels=256, num_heads=8, h=s32, w=s32, p_sizes=(4, 8))
        self.stage2 = HybridModule(in_channels=256, out_channels=128, num_heads=4, h=s16, w=s16, p_sizes=(2, 4))
        self.stage3 = HybridModule(in_channels=128, out_channels=64,  num_heads=2, h=s8,  w=s8,  p_sizes=(1, 2))


    def forward(self, encoder_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        f4 = encoder_features['f4']
        d3 = self.stage1(f4)
        d2 = self.stage2(d3)
        d1 = self.stage3(d2)
        return {'f1': d1, 'f2': d2, 'f3': d3}

class HeteroAE(nn.Module):
    def __init__(self, input_size: int = 256):
        super(HeteroAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(image_size=input_size)

    def forward(self, x: torch.Tensor) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        encoder_features = self.encoder(x)
        decoder_features = self.decoder(encoder_features)
        return encoder_features, decoder_features