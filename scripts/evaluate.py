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


class FeatureComparisonLoss(nn.Module):
    def __init__(self, alpha: float = 0.7):
        super(FeatureComparisonLoss, self).__init__()
        self.alpha = alpha

    def forward(self, features_enc: Dict[str, torch.Tensor], features_dec: Dict[str, torch.Tensor]) -> torch.Tensor:
        # compute per-sample stage losses, then average over stages and batch
        device = next(iter(features_enc.values())).device
        batch_losses = None  # will be a tensor of shape (B,)

        stage_losses = []
        for key in features_dec:
            f_enc, f_dec = features_enc[key], features_dec[key]  # shapes: (B, C, H, W)
            # Per-pixel cosine similarity (B, H, W)
            cos_map = F.cosine_similarity(f_enc, f_dec, dim=1)  # value in [-1,1]

            # Per-pixel MSE across channels (B, H, W) -- match "MSE(FkE(h,w),FkD(h,w))"
            mse_map = torch.mean((f_enc - f_dec) ** 2, dim=1)  # mean across channel dim

            # Per-pixel L_k(h,w) = -alpha * cos + (1-alpha) * mse
            Lk_map = -self.alpha * cos_map + (1.0 - self.alpha) * mse_map  # (B, H, W)

            # Average over spatial dims -> per-sample scalar (B,)
            Lk_per_sample = Lk_map.view(Lk_map.size(0), -1).mean(dim=1)  # (B,)

            stage_losses.append(Lk_per_sample)

        # Sum stages and average -> per-sample loss
        total_per_sample = sum(stage_losses)

        # Finally average over batch -> scalar
        return total_per_sample.mean()


def calculate_anomaly_map(features_enc: Dict[str, torch.Tensor], features_dec: Dict[str, torch.Tensor],
                          input_size: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    anomaly_map = torch.zeros(features_enc['f1'].shape[0], 1, input_size[0], input_size[1]).to(features_enc['f1'].device)
    for key in features_dec:
        f_enc, f_dec = features_enc[key], features_dec[key]
        m_k = 1 - F.cosine_similarity(f_enc, f_dec, dim=1).unsqueeze(1)
        m_k_resized = F.interpolate(m_k, size=input_size, mode='bilinear', align_corners=False)
        anomaly_map += m_k_resized
    anomaly_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
    return anomaly_map, anomaly_score

