import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score, roc_curve
from tqdm import tqdm
import os

from scripts import evaluate


def calculate_specificity(y_true, y_pred):
    """
    Calculates the specificity from the confusion matrix.
    Specificity = True Negatives / (True Negatives + False Positives)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (tn + fp) == 0:
        return 1.0
    return tn / (tn + fp)


def evaluate_anomaly_detector(model, normal_loader, abnormal_loader, device, input_size):
    """
    Evaluates an anomaly detection model.

    Args:
        model (torch.nn.Module): The trained HeteroAE model.
        normal_loader (DataLoader): DataLoader for normal test images.
        abnormal_loader (DataLoader): DataLoader for abnormal test images.
        device (torch.device): The device to run evaluation on.
        input_size (int): The image input size (e.g., 256).
    """
    model.eval()
    all_scores = []
    all_labels = []

    # --- Process Normal Data (Label = 0) ---
    with torch.no_grad():
        for images in tqdm(normal_loader, desc="Evaluating Normal Data"):
            images = images.to(device)
            enc_feats, dec_feats = model(images)
            _, anomaly_score = evaluate.calculate_anomaly_map(enc_feats, dec_feats, input_size=(input_size, input_size))

            all_scores.extend(anomaly_score.cpu().numpy())
            all_labels.extend([0] * images.size(0))

    # --- Process Abnormal Data (Label = 1) ---
    with torch.no_grad():
        for images in tqdm(abnormal_loader, desc="Evaluating Abnormal Data"):
            images = images.to(device)
            enc_feats, dec_feats = model(images)
            _, anomaly_score = evaluate.calculate_anomaly_map(enc_feats, dec_feats, input_size=(input_size, input_size))

            all_scores.extend(anomaly_score.cpu().numpy())
            all_labels.extend([1] * images.size(0))

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)

    auc = roc_auc_score(y_true, y_scores)

    # 2. Find the optimal threshold to convert scores to binary predictions
    # --- Choose threshold that maximizes F1 (paper) ---
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

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = calculate_specificity(y_true, y_pred)

    print("\n--- Anomaly Detection Evaluation Results ---")
    print(f"Optimal Threshold Found: {best_threshold:.4f}")
    print("------------------------------------------")
    print(f"✅ AUC:                  {auc:.4f}")
    print(f"✅ Accuracy:             {accuracy:.4f}")
    print(f"✅ F1-Score:             {f1:.4f}")
    print(f"✅ Sensitivity (Recall): {sensitivity:.4f}")
    print(f"✅ Specificity:          {specificity:.4f}")
    print("------------------------------------------\n")

    return {
        'auc': auc,
        'accuracy': accuracy,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'threshold': best_threshold
    }

