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
from typing import List, Dict, Tuple
import math

def normalize_features(features: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Normalizes a list of feature maps along the channel dimension.
    """
    normalized = []
    for feat in features:
        # Normalize along the channel dimension (C) for each pixel
        norm_feat = F.normalize(feat, p=2, dim=1)
        normalized.append(norm_feat)
    return normalized

class FeatureComparisonLoss(nn.Module):
    def __init__(self, alpha: float = 0.7):
        super(FeatureComparisonLoss, self).__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        self.alpha = alpha
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, encoder_features: List[torch.Tensor], decoder_features: List[torch.Tensor]) -> torch.Tensor:
        if len(encoder_features) != len(decoder_features):
            raise ValueError("Encoder and decoder must have the same number of feature stages.")

        # --- FIX: Normalize features before loss calculation for consistency ---
        norm_encoder_features = normalize_features(encoder_features)
        norm_decoder_features = normalize_features(decoder_features)

        total_fc_loss = 0.0
        for k, (enc_feat, dec_feat) in enumerate(zip(norm_encoder_features, norm_decoder_features)):
            if enc_feat.shape != dec_feat.shape:
                raise ValueError(f"Shape mismatch in stage {k}: encoder {enc_feat.shape}, decoder {dec_feat.shape}")

            cos_sim = self.cosine_similarity(enc_feat, dec_feat)
            mse = self.mse_loss(enc_feat, dec_feat).mean(dim=1)
            loss_k_map = -self.alpha * cos_sim + (1 - self.alpha) * mse
            stage_loss = torch.mean(loss_k_map)
            total_fc_loss += stage_loss
        return total_fc_loss

def calculate_anomaly_map(encoder_features: List[torch.Tensor], decoder_features: List[torch.Tensor], input_image_size: Tuple[int, int]) -> torch.Tensor:
    if len(encoder_features) != len(decoder_features):
        raise ValueError("Encoder and decoder must have the same number of feature stages.")

    # --- FIX: Move normalization inside the function to ensure it's always applied ---
    norm_encoder_features = normalize_features(encoder_features)
    norm_decoder_features = normalize_features(decoder_features)

    batch_size = norm_encoder_features[0].size(0)
    device = norm_encoder_features[0].device
    final_anomaly_map = torch.zeros(batch_size, 1, *input_image_size, device=device)
    cosine_similarity = nn.CosineSimilarity(dim=1)

    for enc_feat, dec_feat in zip(norm_encoder_features, norm_decoder_features):
        anomaly_map_k = 1 - cosine_similarity(enc_feat, dec_feat)
        anomaly_map_k = anomaly_map_k.unsqueeze(1)
        resized_anomaly_map_k = F.interpolate(anomaly_map_k, size=input_image_size, mode='bilinear', align_corners=False)
        final_anomaly_map += resized_anomaly_map_k # --- FIX: Changed from average to sum ---

    return final_anomaly_map

def get_anomaly_score(anomaly_map: torch.Tensor) -> torch.Tensor:
    batch_size = anomaly_map.size(0)
    flattened_map = anomaly_map.view(batch_size, -1)
    anomaly_scores, _ = torch.max(flattened_map, dim=1)
    return anomaly_scores