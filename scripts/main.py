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
import sys

np.random.seed(42)
import train
import evaluate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models
from data import data_loader
from utils import visualization, metrics

DATA_DIR = 'chest_xray/chest_xray'
INPUT_SIZE = 256
BATCH_SIZE = 200
LEARNING_RATE = 1e-4
EPOCHS = 200
ALPHA_LOSS = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader, test_normal_loader, test_abnormal_loader = data_loader.get_dataloader(INPUT_SIZE, BATCH_SIZE)

best_model = train.train(
        train_loader, test_normal_loader, test_abnormal_loader,
        EPOCHS, device, ALPHA_LOSS, LEARNING_RATE, INPUT_SIZE
)

metrics.evaluate_anomaly_detector(
        model=best_model,
        normal_loader=test_normal_loader,
        abnormal_loader=test_abnormal_loader,
        device=device,
        input_size=INPUT_SIZE
)

