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
from sklearn.metrics import roc_curve, accuracy_score, f1_score

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

    # Variables for tracking the best model and early stopping
    best_accuracy = 0.0
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
            loss = loss_fn(encoder_features, decoder_features)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_train_loss:.4f}")

        # --- Start of Validation Phase ---
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            # Process normal (label 0) test images
            for images in tqdm(test_normal_loader, desc="Evaluating Normal"):
                images = images.to(device)
                enc_feats, dec_feats = model(images)
                _, anomaly_score = evaluate.calculate_anomaly_map(enc_feats, dec_feats,
                                                                  input_size=(input_size, input_size))
                all_scores.extend(anomaly_score.cpu().numpy())
                all_labels.extend([0] * images.size(0))

            # Process abnormal (label 1) test images
            for images in tqdm(test_abnormal_loader, desc="Evaluating Abnormal"):
                images = images.to(device)
                enc_feats, dec_feats = model(images)
                _, anomaly_score = evaluate.calculate_anomaly_map(enc_feats, dec_feats,
                                                                  input_size=(input_size, input_size))
                all_scores.extend(anomaly_score.cpu().numpy())
                all_labels.extend([1] * images.size(0))

        y_true = np.array(all_labels)
        y_scores = np.array(all_scores)

        # Find the best threshold to maximize validation accuracy
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        # compute F1 at each candidate threshold from roc_curve
        f1_list = []
        for thr in roc_thresholds:
            y_pred_thr = (y_scores >= thr).astype(int)
            # zero_division=0 avoids errors when there are no positives predicted
            f1_list.append(f1_score(y_true, y_pred_thr, zero_division=0))

        best_idx = int(np.argmax(f1_list))
        best_threshold = float(roc_thresholds[best_idx])

        # Use that threshold for binary predictions
        y_pred = (y_scores >= best_threshold).astype(int)

        epoch_accuracy = accuracy_score(y_true, y_pred)

        print(f"Epoch {epoch + 1} Validation Accuracy: {epoch_accuracy:.4f}")

        # --- Best Model Selection and Early Stopping Logic ---
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            epochs_no_improve = 0
            # Save the state of the best model found so far
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"🎉 New best model found with Accuracy: {best_accuracy:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement.")
            break

    # --- End of Training Loop ---
    print("\n--- Training Finished ---")

    # Load the best model state before returning
    if best_model_state:
        print(f"Loading best model with Accuracy: {best_accuracy:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("Warning: Training finished without a best model state being saved.")

    return model

