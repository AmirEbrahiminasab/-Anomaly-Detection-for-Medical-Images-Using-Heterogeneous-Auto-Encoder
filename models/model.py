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
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.stage2 = resnet.layer1
        self.stage3 = resnet.layer2
        self.stage4 = resnet.layer3
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
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
    def __init__(self, in_channels: int, num_heads: int, h: int, w: int, p1_size: int, p2_size: int):
        super(MSTB, self).__init__()
        assert num_heads % 2 == 0, "num_heads must be even because we split heads across p1/p2"
        assert h % p1_size == 0 and w % p1_size == 0, "H,W must be divisible by p1_size"
        assert h % p2_size == 0 and w % p2_size == 0, "H,W must be divisible by p2_size"

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        num_patches_local = h * w
        num_patches_p1 = (h // p1_size) * (w // p1_size)
        num_patches_p2 = (h // p2_size) * (w // p2_size)

        self.pos_embed_local = nn.Parameter(torch.zeros(1, num_patches_local, in_channels))
        self.pos_embed_p1 = nn.Parameter(torch.zeros(1, num_patches_p1, in_channels))
        self.pos_embed_p2 = nn.Parameter(torch.zeros(1, num_patches_p2, in_channels))
        trunc_normal_(self.pos_embed_local, std=.02)
        trunc_normal_(self.pos_embed_p1, std=.02)
        trunc_normal_(self.pos_embed_p2, std=.02)

        # patch embedding convs (non-overlapping)
        self.patch_embed_p1 = nn.Conv2d(in_channels, in_channels, kernel_size=p1_size, stride=p1_size)
        self.patch_embed_p2 = nn.Conv2d(in_channels, in_channels, kernel_size=p2_size, stride=p2_size)

        # projections
        self.to_q = nn.Linear(in_channels, in_channels)
        self.to_k = nn.Linear(in_channels, in_channels)
        self.to_v = nn.Linear(in_channels, in_channels)
        self.to_out = nn.Linear(in_channels, in_channels)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 4 * in_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * in_channels, in_channels),
            nn.Dropout(0.1),
        )
        self.norm_q = nn.LayerNorm(in_channels)   # normalize queries as in pre-norm design
        self.norm2 = nn.LayerNorm(in_channels)    # LN applied to Z before MLP
        self.norm_out = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # local tokens
        q_local = x.flatten(2).transpose(1, 2)                # (B, H*W, C)
        q_local = q_local + self.pos_embed_local             # add pos emb
        q_norm = self.norm_q(q_local)                        # LN before Q projection
        q_proj = self.to_q(q_norm)                           # (B, H*W, C)

        # reshape to heads: (B, num_heads, H*W, head_dim)
        q = q_proj.reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q1, q2 = torch.chunk(q, 2, dim=1)   # split heads into two groups

        # --- regional scale p1 ---
        x_p1 = self.patch_embed_p1(x).flatten(2).transpose(1, 2) + self.pos_embed_p1  # (B, Np1, C)
        k_p1 = self.to_k(x_p1).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_p1 = self.to_v(x_p1).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k1, _ = torch.chunk(k_p1, 2, dim=1)
        v1, _ = torch.chunk(v_p1, 2, dim=1)

        # --- regional scale p2 ---
        x_p2 = self.patch_embed_p2(x).flatten(2).transpose(1, 2) + self.pos_embed_p2  # (B, Np2, C)
        k_p2 = self.to_k(x_p2).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_p2 = self.to_v(x_p2).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        _, k2 = torch.chunk(k_p2, 2, dim=1)
        _, v2 = torch.chunk(v_p2, 2, dim=1)

        # --- attention p1 ---
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        out1 = (attn1 @ v1).transpose(1, 2).reshape(B, H * W, C // 2)

        # --- attention p2 ---
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        out2 = (attn2 @ v2).transpose(1, 2).reshape(B, H * W, C // 2)

        # concat multi-scale outputs -> Z (B, H*W, C)
        Z = torch.cat([out1, out2], dim=-1)

        # project and then follow paper: Z -> LN -> MLP (with residual)
        Z = self.norm_out(Z)          # normalize concatenated [out1,out2]
        Z = self.to_out(Z)           # final linear projection
        Z = Z + q_local              # residual with local information (keeps local detail)
        Z = Z + self.mlp(self.norm2(Z))

        # back to spatial shape
        return Z.transpose(1, 2).reshape(B, C, H, W)


class HybridModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int, h: int, w: int, p_sizes: tuple):
        super(HybridModule, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
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
    def __init__(self, input_size: int = 256):
        super(Decoder, self).__init__()
        s = input_size
        h_s1, w_s1 = s // 4, s // 4
        h_s2, w_s2 = s // 8, s // 8
        h_s3, w_s3 = s // 16, s // 16

        # --- Decoder stage instantiation (correct p_sizes mapping per paper) ---
        # stage1 is the top (largest spatial resolution): use p1=4, p2=8
        self.stage1 = HybridModule(in_channels=64, out_channels=64, num_heads=2,
                                   h=h_s1, w=w_s1, p_sizes=(4, 8))

        # stage2 middle resolution: use p1=2, p2=4
        self.stage2 = HybridModule(in_channels=128, out_channels=64, num_heads=4,
                                   h=h_s2, w=w_s2, p_sizes=(2, 4))

        # stage3 deepest (bottleneck -> upsample to mid): use p1=1, p2=2
        self.stage3 = HybridModule(in_channels=256, out_channels=128, num_heads=8,
                                   h=h_s3, w=w_s3, p_sizes=(1, 2))

        # keep the final stage1 upsample = Identity so final f1_dec matches encoder F1 spatial size
        self.stage1.upsample = nn.Identity()

    def forward(self, f4: torch.Tensor) -> dict:
        f3_dec = self.stage3(f4)
        f2_dec = self.stage2(f3_dec)
        f1_dec = self.stage1(f2_dec)
        return {'f3': f3_dec, 'f2': f2_dec, 'f1': f1_dec}


# --- 3. Full Hetero-AE Model ---
class HeteroAE(nn.Module):
    def __init__(self, input_size: int = 256):
        super(HeteroAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(input_size=input_size)

    def forward(self, x: torch.Tensor) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        encoder_features = self.encoder(x)
        decoder_features = self.decoder(encoder_features['f4'])
        return encoder_features, decoder_features

