import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import models
from typing import List, Dict
import math

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models
from scripts import evaluate


def train(train_loader, test_normal_loader, test_abnormal_loader, epochs, device, alpha_loss, learning_rate, input_size):
    """Function to define the model"""
    model = models.model.HeteroAE(input_size=input_size).to(device)
    loss_fn = evaluate.FeatureComparisonLoss(alpha=alpha_loss).to(device)
    # optimizer = torch.optim.Adam(model.decoder.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images in progress_bar:
            images = images.to(device)
            optimizer.zero_grad()
            encoder_features, decoder_features = model(images)
            loss = loss_fn(encoder_features, decoder_features)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_train_loss:.4f}")

        # --- Evaluation Loop ---
        model.eval()
        normal_scores = []
        abnormal_scores = []
        with torch.no_grad():
            for images in tqdm(test_normal_loader, desc="Evaluating Normal"):
                images = images.to(device)
                enc_feats, dec_feats = model(images)
                _, anomaly_score = evaluate.calculate_anomaly_map(enc_feats, dec_feats,
                                                         input_size=(input_size, input_size))
                normal_scores.extend(anomaly_score.cpu().numpy())
            for images in tqdm(test_abnormal_loader, desc="Evaluating Abnormal"):
                images = images.to(device)
                enc_feats, dec_feats = model(images)
                _, anomaly_score = evaluate.calculate_anomaly_map(enc_feats, dec_feats,
                                                         input_size=(input_size, input_size))
                abnormal_scores.extend(anomaly_score.cpu().numpy())

        avg_normal_score = np.mean(normal_scores)
        avg_abnormal_score = np.mean(abnormal_scores)

        print(f"Epoch {epoch + 1} Evaluation:")
        print(f"  - Average Anomaly Score (Normal):   {avg_normal_score:.4f}")
        print(f"  - Average Anomaly Score (Abnormal): {avg_abnormal_score:.4f}")

    print("---Training Finished---")

    return model
