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
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for loader, label in [(normal_loader, 0), (abnormal_loader, 1)]:
            desc = "Final Eval: Normal" if label == 0 else "Final Eval: Abnormal"
            for images in tqdm(loader, desc=desc):
                images = images.to(device)
                enc_feats, dec_feats = model(images)

                enc_for_map = [enc_feats['f1'], enc_feats['f2'], enc_feats['f3']]
                dec_for_map = [dec_feats['f1'], dec_feats['f2'], dec_feats['f3']]

                # Normalization is handled inside calculate_anomaly_map
                anomaly_map = evaluate.calculate_anomaly_map(enc_for_map, dec_for_map, (input_size, input_size))

                anomaly_score = evaluate.get_anomaly_score(anomaly_map)
                all_scores.extend(anomaly_score.cpu().numpy())
                all_labels.extend([label] * images.size(0))

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

    # Find the best threshold based on F1-score
    f1_list = [f1_score(y_true, (y_scores >= thr).astype(int), zero_division=0) for thr in roc_thresholds]
    best_idx = int(np.argmax(f1_list))
    best_threshold = float(roc_thresholds[best_idx])

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
    return {'auc': auc, 'accuracy': accuracy, 'f1_score': f1, 'sensitivity': sensitivity, 'specificity': specificity, 'threshold': best_threshold}

