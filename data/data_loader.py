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
from typing import List, Dict, Optional, Tuple
import math
from PIL import Image
from scipy.io import loadmat
import h5py


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        # self.image_paths = glob.glob(os.path.join(root_dir, '*.jpeg'))
        self.image_paths = []
        for ext in ('*.jpeg','*.jpg','*.png'):
            self.image_paths += glob.glob(os.path.join(root_dir, ext))
        self.image_paths = sorted(self.image_paths)
        self.transform = transform
        if not self.image_paths:
            raise RuntimeError(f"No .jpeg images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_dataloader_chest_xray(input_size=256, batch_size=16):
    DATA_DIR = "/home/appliedailab/Desktop/Deep/-Anomaly-Detection-for-Medical-Images-Using-Heterogeneous-Auto-Encoder/data/dataset/chest_xray/chest_xray"
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # --- Datasets and DataLoaders ---
    train_normal_dir = os.path.join(DATA_DIR, 'train', 'NORMAL')
    train_dataset = ChestXRayDataset(root_dir=train_normal_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    test_normal_dir = os.path.join(DATA_DIR, 'test', 'NORMAL')
    test_abnormal_dir = os.path.join(DATA_DIR, 'test', 'PNEUMONIA')

    test_normal_dataset = ChestXRayDataset(root_dir=test_normal_dir, transform=test_transform)
    test_abnormal_dataset = ChestXRayDataset(root_dir=test_abnormal_dir, transform=test_transform)

    test_normal_loader = DataLoader(test_normal_dataset, batch_size=batch_size, shuffle=False)
    test_abnormal_loader = DataLoader(test_abnormal_dataset, batch_size=batch_size, shuffle=False)

    print(f"Found {len(train_dataset)} normal images for training.")
    print(f"Found {len(test_normal_dataset)} normal images for testing.")
    print(f"Found {len(test_abnormal_dataset)} abnormal images for testing.")
    
    return train_loader, test_normal_loader, test_abnormal_loader


import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import h5py

class BrainTumorDatasetH5py(Dataset):
    """
    Dataset class for the Brain Tumor dataset, loading data from .mat files using h5py.
    Handles conversion from grayscale to 3-channel RGB.
    """
    def __init__(self, mat_file_paths: list, transform=None):
        """
        Args:
            mat_file_paths (list): List of paths to the .mat files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mat_file_paths = sorted(mat_file_paths)
        self.transform = transform
        if not self.mat_file_paths:
            raise RuntimeError("No .mat files found in the provided list of paths.")

    def __len__(self):
        return len(self.mat_file_paths)

    def __getitem__(self, idx):
        mat_path = self.mat_file_paths[idx]
        with h5py.File(mat_path, 'r') as f:
            # Access the image data within the HDF5 file structure
            # The [()] syntax loads the data into a NumPy array
            image_data = f['cjdata/image'][()].astype(np.float32)

        # Normalize the image to 0-255 range, mimicking the original MATLAB script
        min_val = image_data.min()
        max_val = image_data.max()
        if max_val > min_val:
            image_data = (image_data - min_val) / (max_val - min_val) * 255.0
        
        image_data = image_data.astype(np.uint8)

        # Convert grayscale NumPy array to a 3-channel PIL Image
        # The .convert('RGB') is crucial for your model's requirement
        image = Image.fromarray(image_data).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image

def get_dataloader_brain_tumor(dataset_root='', input_size=256, batch_size=16):
    """
    Creates and returns DataLoaders for the brain tumor dataset using h5py.

    Assumes:
    - Label 1 (Meningioma) is the 'normal' class for anomaly detection training.
    - Labels 2 (Glioma) and 3 (Pituitary) are 'abnormal' classes for testing.
    """
    # --- IMPORTANT: Update this path to your brain tumor dataset directory ---
    DATA_DIR = dataset_root
    
    # Standard ImageNet normalization for 3-channel images
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # --- Find all .mat files and sort them by label using h5py ---
    all_mat_files = glob.glob(os.path.join(DATA_DIR, '**', '*.mat'), recursive=True)
    
    normal_files = []
    abnormal_files = []

    print("Sorting files by label... this may take a moment.")
    for mat_path in all_mat_files:
        try:
            with h5py.File(mat_path, 'r') as f:
                # Access the scalar label and convert to a Python integer
                label = int(f['cjdata/label'][()])
            
            if label == 1:  # Meningioma is 'normal'
                normal_files.append(mat_path)
            else:  # Glioma (2) and Pituitary (3) are 'abnormal'
                abnormal_files.append(mat_path)
        except Exception as e:
            print(f"Could not process file {mat_path}: {e}")

    # --- Split normal data into training and testing sets (80/20 split) ---
    np.random.shuffle(normal_files)  # Shuffle for a random split
    split_idx = int(len(normal_files) * 0.8)
    train_normal_paths = normal_files[:split_idx]
    test_normal_paths = normal_files[split_idx:]

    # --- Datasets and DataLoaders ---
    train_dataset = BrainTumorDatasetH5py(mat_file_paths=train_normal_paths, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    test_normal_dataset = BrainTumorDatasetH5py(mat_file_paths=test_normal_paths, transform=test_transform)
    test_abnormal_dataset = BrainTumorDatasetH5py(mat_file_paths=abnormal_files, transform=test_transform)

    test_normal_loader = DataLoader(test_normal_dataset, batch_size=batch_size, shuffle=False)
    test_abnormal_loader = DataLoader(test_abnormal_dataset, batch_size=batch_size, shuffle=False)

    print("-" * 30)
    print(f"Found {len(train_dataset)} normal images for training.")
    print(f"Found {len(test_normal_dataset)} normal images for testing.")
    print(f"Found {len(test_abnormal_dataset)} abnormal images for testing.")
    print("-" * 30)
    
    return train_loader, test_normal_loader, test_abnormal_loader


# ──────────────────────────────────────────────────────────────────────────────
# COVID dataset
# Train on:  <root>/Train/Normal
# Test on:   <root>/Val/Normal  vs  <root>/Val/Covid
# ──────────────────────────────────────────────────────────────────────────────
import os, glob
from typing import List
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Covid19_Dataset(Dataset):
    """Image-only dataset over a directory of images (jpg/png/jpeg)."""
    def __init__(self, root_dir: str, transform=None):
        self.image_paths: List[str] = []
        for ext in ('*.jpeg', '*.jpg', '*.png'):
            self.image_paths += glob.glob(os.path.join(root_dir, ext))
        self.image_paths = sorted(self.image_paths)
        self.transform = transform
        if not self.image_paths:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def _find_subdir_ci(base: str, name: str) -> str:
    """Find a subdir under base, case-insensitive (returns first match)."""
    candidates = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    for d in candidates:
        if d.lower() == name.lower():
            return os.path.join(base, d)
    # also allow partial (e.g., 'train' matches 'Training' if needed)
    for d in candidates:
        if name.lower() in d.lower():
            return os.path.join(base, d)
    raise FileNotFoundError(f"Expected subdir '{name}' under {base}")

def get_dataloader_covid19(
    dataset_root: str,
    input_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 2,
):
    """
    Returns:
        train_loader, test_normal_loader, test_abnormal_loader
    Layout (case-insensitive):
        <root>/Train/Normal, <root>/Train/Covid
        <root>/Val/Normal,   <root>/Val/Covid
    Training uses only Normal from Train; testing uses Val/Normal vs Val/Covid.
    """
    # Folders (case-insensitive resolution)
    train_dir = _find_subdir_ci(dataset_root, "Train")
    val_dir   = _find_subdir_ci(dataset_root, "Val")

    train_normal_dir = _find_subdir_ci(train_dir, "Normal")
    val_normal_dir   = _find_subdir_ci(val_dir, "Normal")
    val_covid_dir    = _find_subdir_ci(val_dir, "Covid")

    # Transforms (match your other loaders)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    test_tfm = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Datasets & Loaders (train on Normal; test on Normal vs Covid)
    train_dataset        = Covid19_Dataset(train_normal_dir, transform=train_tfm)
    test_normal_dataset  = Covid19_Dataset(val_normal_dir,   transform=test_tfm)
    test_abnormal_dataset= Covid19_Dataset(val_covid_dir,    transform=test_tfm)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_normal_loader = DataLoader(test_normal_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)
    test_abnormal_loader = DataLoader(test_abnormal_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)

    print("-" * 30)
    print(f"[COVID] Train (Normal): {len(train_dataset)}")
    print(f"[COVID] Test  (Normal): {len(test_normal_dataset)}")
    print(f"[COVID] Test  (Covid):  {len(test_abnormal_dataset)}")
    print("-" * 30)

    return train_loader, test_normal_loader, test_abnormal_loader
