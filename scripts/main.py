import torch
import os
import numpy as np
import sys

np.random.seed(42)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train import train
import models
from data.data_loader import get_dataloader_brain_tumor, get_dataloader_chest_xray, get_dataloader_covid19
from utils.metrics import calculate_specificity, evaluate_anomaly_detector

INPUT_SIZE = 256
BATCH_SIZE = 200
LEARNING_RATE = 1e-4
EPOCHS = 200
ALPHA_LOSS = 0.7


def chest_xray():
    DATA_DIR = 'chest_xray/chest_xray'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_chest_xray(INPUT_SIZE, BATCH_SIZE)

    best_model = train(
        train_loader, test_normal_loader, test_abnormal_loader,
        EPOCHS, device, ALPHA_LOSS, LEARNING_RATE, INPUT_SIZE, patience=500
    )

    SAVE_DIR = 'models/saved_models'
    SAVE_PATH = os.path.join(SAVE_DIR, 'best_model_weights_chest.pth')
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(best_model.state_dict(), SAVE_PATH)
    print(f"✅ Best model's weights saved to: {SAVE_PATH}")

    evaluate_anomaly_detector(
        model=best_model,
        normal_loader=test_normal_loader,
        abnormal_loader=test_abnormal_loader,
        device=device,
        input_size=INPUT_SIZE
    )


def brain_tumor():
    DATA_DIR = "/home/appliedailab/Desktop/Deep/brain_tumor"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_brain_tumor(
        dataset_root=DATA_DIR,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE
    )

    best_model = train(
        train_loader, test_normal_loader, test_abnormal_loader,
        EPOCHS, device, ALPHA_LOSS, LEARNING_RATE, INPUT_SIZE, patience=500
    )

    SAVE_DIR = 'models/saved_models'
    SAVE_PATH = os.path.join(SAVE_DIR, 'best_model_weights_brain.pth')
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(best_model.state_dict(), SAVE_PATH)
    print(f"✅ Best model's weights saved to: {SAVE_PATH}")

    evaluate_anomaly_detector(
        model=best_model,
        normal_loader=test_normal_loader,
        abnormal_loader=test_abnormal_loader,
        device=device,
        input_size=INPUT_SIZE
    )


def covid19():
    # DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CovidDataset"))
    DATA_DIR = "/home/appliedailab/Desktop/Deep/CovidDataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.__version__)

    train_loader, test_normal_loader, test_abnormal_loader = get_dataloader_covid19(
        dataset_root=DATA_DIR,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        download_url="https://figshare.com/ndownloader/files/50920287",  # used only if folder missing
    )

    best_model = train(
        train_loader, test_normal_loader, test_abnormal_loader,
        EPOCHS, device, ALPHA_LOSS, LEARNING_RATE, INPUT_SIZE, patience=500
    )

    SAVE_DIR = 'models/saved_models'
    os.makedirs(SAVE_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(SAVE_DIR, 'best_model_weights_covid19.pth')
    torch.save(best_model.state_dict(), SAVE_PATH)
    print(f"Best model's weights saved to: {SAVE_PATH}")

    evaluate_anomaly_detector(
        model=best_model,
        normal_loader=test_normal_loader,
        abnormal_loader=test_abnormal_loader,
        device=device,
        input_size=INPUT_SIZE
    )


if __name__ == "__main__":
    chest_xray()
    brain_tumor()
    covid19()

