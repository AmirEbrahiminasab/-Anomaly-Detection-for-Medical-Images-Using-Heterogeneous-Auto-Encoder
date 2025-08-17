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
import shutil
import tarfile
import tempfile
import zipfile
from urllib.request import urlopen, Request

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

def _download_file(url: str, dst_path: str):
    """Download with urllib (handles redirects)."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(dst_path, "wb") as f:
        shutil.copyfileobj(r, f)

def _extract_archive(archive_path: str, extract_to: str):
    os.makedirs(extract_to, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_to)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_to)
    else:
        raise RuntimeError(f"Unsupported archive: {archive_path}")

def _flatten_single_topdir(src_root: str, dst_root: str):
    """
    If src_root has exactly one top-level dir, move its contents into dst_root.
    """
    entries = [d for d in os.listdir(src_root) if not d.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(src_root, entries[0])):
        inner = os.path.join(src_root, entries[0])
        for name in os.listdir(inner):
            src = os.path.join(inner, name)
            dst = os.path.join(dst_root, name)
            if os.path.exists(dst):
                # merge folders; overwrite files
                if os.path.isdir(src):
                    for base, dirs, files in os.walk(src):
                        rel = os.path.relpath(base, src)
                        outdir = os.path.join(dst, rel)
                        os.makedirs(outdir, exist_ok=True)
                        for f in files:
                            shutil.copy2(os.path.join(base, f), os.path.join(outdir, f))
                else:
                    shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)

def _ensure_download_if_missing(dataset_root: str, download_url: Optional[str]):
    """Download & extract only if the folder itself doesn't exist."""
    if os.path.isdir(dataset_root):
        return  # <-- do nothing; don't rely on inner contents

    if not download_url:
        # create empty dir so downstream code can continue without validation
        os.makedirs(dataset_root, exist_ok=True)
        print(f"[COVID] Created empty dataset root at: {dataset_root} (no download_url provided)")
        return

    os.makedirs(dataset_root, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = os.path.join(tmp, "covid_dataset_archive")
        print(f"[COVID] Downloading dataset (first-time setup):\n  {download_url}")
        _download_file(download_url, archive_path)
        print(f"[COVID] Extracting ...")
        extract_here = os.path.join(tmp, "extract")
        _extract_archive(archive_path, extract_here)
        _flatten_single_topdir(extract_here, dataset_root)
    print(f"[COVID] Dataset prepared at: {dataset_root}")

class ListImageDataset(Dataset):
    def __init__(self, image_paths: List[str], transform=None, allow_empty: bool=False):
        if not image_paths and not allow_empty:
            raise RuntimeError("ListImageDataset got an empty list.")
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img

def _list_images_recursive(root_dir: str, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    """Recursively list image files (case-insensitive extensions)."""
    exts = tuple(e.lower() for e in exts)
    out = []
    for base, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(base, f))
    return sorted(out)

# ==== helper to find sibling chest_xray normals (case-insensitive) ====
def _find_ci(base: str, name: str) -> Optional[str]:
    if not os.path.isdir(base): return None
    for d in os.listdir(base):
        p = os.path.join(base, d)
        if os.path.isdir(p) and d.lower() == name.lower():
            return p
    for d in os.listdir(base):
        p = os.path.join(base, d)
        if os.path.isdir(p) and name.lower() in d.lower():
            return p
    return None

def _discover_chest_xray_normals(repo_root: str) -> Tuple[List[str], List[str]]:
    """
    Return (train_normals, val_normals) image lists from ../chest_xray/{train,val}/NORMAL.
    """
    cx_root = _find_ci(repo_root, "chest_xray")
    if not cx_root: return [], []
    train_dir = _find_ci(cx_root, "train") or _find_ci(cx_root, "Train")
    val_dir   = _find_ci(cx_root, "val")   or _find_ci(cx_root, "validation") or _find_ci(cx_root, "Val")
    if not train_dir or not val_dir: return [], []
    train_normal = _find_ci(train_dir, "NORMAL") or _find_ci(train_dir, "Normal")
    val_normal   = _find_ci(val_dir,   "NORMAL") or _find_ci(val_dir,   "Normal")
    if not train_normal or not val_normal: return [], []
    return _list_images_recursive(train_normal), _list_images_recursive(val_normal)

def get_dataloader_covid19(
    dataset_root: str,
    input_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 2,
    download_url: Optional[str] = None,
    normals_dir: Optional[str] = None,
    covid_val_ratio: float = 1.0,
):
    # Only download if the folder doesn't exist. Do not inspect contents.
    _ensure_download_if_missing(dataset_root, download_url)

    # Collect COVID images (may be empty; that’s fine)
    covid_imgs = _list_images_recursive(dataset_root)

    # Get NORMALs (from sibling chest_xray by default)
    repo_root = os.path.abspath(os.path.join(dataset_root, ".."))
    if normals_dir:
        normal_train_imgs = _list_images_recursive(os.path.join(normals_dir, "train")) + \
                            _list_images_recursive(os.path.join(normals_dir, "Train")) or \
                            _list_images_recursive(normals_dir)
        normal_val_imgs   = _list_images_recursive(os.path.join(normals_dir, "val")) + \
                            _list_images_recursive(os.path.join(normals_dir, "Val"))
    else:
        normal_train_imgs, normal_val_imgs = _discover_chest_xray_normals(repo_root)

    if not normal_train_imgs or not normal_val_imgs:
        raise RuntimeError(
            "Could not find NORMAL images.\n"
            "Provide `normals_dir` (root containing train/val NORMAL) OR ensure a sibling 'chest_xray' dataset exists."
        )

    # Split COVID into train/val virtually (we only use val as abnormal set)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(covid_imgs))
    split = int((1.0 - covid_val_ratio) * max(0, len(idx)))
    covid_train = [covid_imgs[i] for i in idx[:split]]
    covid_val   = [covid_imgs[i] for i in idx[split:]]

    # Transforms
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

    # Datasets (allow empty abnormal set; evaluation will simply see 0 batches)
    train_dataset         = ListImageDataset(normal_train_imgs, transform=train_tfm)
    test_normal_dataset   = ListImageDataset(normal_val_imgs,   transform=test_tfm)
    test_abnormal_dataset = ListImageDataset(covid_val,         transform=test_tfm, allow_empty=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_normal_loader = DataLoader(test_normal_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)
    test_abnormal_loader = DataLoader(test_abnormal_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)

    print("-" * 36)
    print(f"[COVID] Train (Normal): {len(train_dataset)}")
    print(f"[COVID] Test  (Normal): {len(test_normal_dataset)}")
    print(f"[COVID] Test  (Covid):  {len(test_abnormal_dataset)}")
    if len(test_abnormal_dataset) == 0:
        print("[WARN] No COVID images found under CovidDataset; abnormal test set is empty.")
    print(f"[COVID] covid_val_ratio={covid_val_ratio:.2f}  (covid_train={len(covid_train)}, covid_val={len(covid_val)})")
    print("-" * 36)

    return train_loader, test_normal_loader, test_abnormal_loader
