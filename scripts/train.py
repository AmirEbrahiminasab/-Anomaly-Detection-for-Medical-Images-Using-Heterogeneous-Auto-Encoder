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
import copy
import random

# Add necessary imports from sklearn for calculating accuracy and roc_curve
from sklearn.metrics import roc_curve, accuracy_score, f1_score, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import model as modell
from scripts import evaluate


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(train_loader, test_normal_loader, test_abnormal_loader, epochs, device, alpha_loss, learning_rate, input_size,
          patience=10):
    """
    Function to train the model with best model selection and early stopping.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping early.
    """
    set_seed(42)

    model = modell.HeteroAE(input_size=input_size).to(device)
    loss_fn = evaluate.FeatureComparisonLoss(alpha=alpha_loss).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Use AUC as the primary metric for saving the best model, as it's better for imbalanced datasets.
    best_auc = 0.0
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images in progress_bar:
            images = images.to(device)
            optimizer.zero_grad()
            encoder_features, decoder_features = model(images)

            enc_for_loss = [encoder_features['f1'], encoder_features['f2'], encoder_features['f3']]
            dec_for_loss = [decoder_features['f1'], decoder_features['f2'], decoder_features['f3']]

            loss = loss_fn(enc_for_loss, dec_for_loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_train_loss:.4f}")

        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for loader, label in [(test_normal_loader, 0), (test_abnormal_loader, 1)]:
                desc = "Evaluating Normal" if label == 0 else "Evaluating Abnormal"
                for images in tqdm(loader, desc=desc):
                    images = images.to(device)
                    enc_feats, dec_feats = model(images)

                    enc_for_map = [enc_feats['f1'], enc_feats['f2'], enc_feats['f3']]
                    dec_for_map = [dec_feats['f1'], dec_feats['f2'], dec_feats['f3']]

                    # Normalization is now handled inside calculate_anomaly_map
                    anomaly_map = evaluate.calculate_anomaly_map(enc_for_map, dec_for_map, (input_size, input_size))

                    anomaly_score = evaluate.get_anomaly_score(anomaly_map)
                    all_scores.extend(anomaly_score.cpu().numpy())
                    all_labels.extend([label] * images.size(0))

        y_true = np.array(all_labels)
        y_scores = np.array(all_scores)

        epoch_auc = roc_auc_score(y_true, y_scores)
        print(f"Epoch {epoch + 1} Validation AUC: {epoch_auc:.4f}")

        if epoch_auc > best_auc:
            best_auc = epoch_auc
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"🎉 New best model found with AUC: {best_auc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement.")
            break

    print("\n--- Training Finished ---")
    if best_model_state:
        print(f"Loading best model with AUC: {best_auc:.4f}")
        model.load_state_dict(best_model_state)
    return model

