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
        for ext in ('*.jpeg', '*.jpg', '*.png'):
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
    DATA_DIR = "/home/appliedailab/Desktop/Deep/chest_xray/chest_xray"
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5),
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
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


class BrainTumorDatasetH5py(Dataset):

    def __init__(self, root_dir: str, transform=None):
        # self.image_paths = glob.glob(os.path.join(root_dir, '*.jpeg'))
        self.image_paths = []
        for ext in ('*.jpeg', '*.jpg', '*.png'):
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


def get_dataloader_brain_tumor(dataset_root='', input_size=256, batch_size=16):
    DATA_DIR = dataset_root
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # --- Datasets and DataLoaders ---
    train_normal_dir = os.path.join(DATA_DIR, 'Training', 'notumor')
    train_dataset = ChestXRayDataset(root_dir=train_normal_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    test_normal_dir = os.path.join(DATA_DIR, 'Testing', 'notumor')
    test_abnormal_dir = os.path.join(DATA_DIR, 'Testing', 'tumor')

    test_normal_dataset = ChestXRayDataset(root_dir=test_normal_dir, transform=test_transform)
    test_abnormal_dataset = ChestXRayDataset(root_dir=test_abnormal_dir, transform=test_transform)

    test_normal_loader = DataLoader(test_normal_dataset, batch_size=batch_size, shuffle=False)
    test_abnormal_loader = DataLoader(test_abnormal_dataset, batch_size=batch_size, shuffle=False)

    print(f"Found {len(train_dataset)} normal images for training.")
    print(f"Found {len(test_normal_dataset)} normal images for testing.")
    print(f"Found {len(test_abnormal_dataset)} abnormal images for testing.")

    return train_loader, test_normal_loader, test_abnormal_loader


# ──────────────────────────────────────────────────────────────────────────────
# COVID-19 loader (no metadata; download only if folder absent)
# Returns: train_loader, test_normal_loader, test_abnormal_loader
# ──────────────────────────────────────────────────────────────────────────────
import os
import shutil
import tempfile
import zipfile
import tarfile
from urllib.request import urlopen, Request
from typing import List, Iterable, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------- helpers ----------

def _download_file(url: str, dst_path: str):
    """Downloads a file from a URL to a destination path."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(dst_path, "wb") as f:
        shutil.copyfileobj(r, f)


def _extract_archive(archive_path: str, extract_to: str):
    """Extracts a zip or tar archive to a specified directory."""
    os.makedirs(extract_to, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_to)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(extract_to)
    else:
        raise RuntimeError(f"Unsupported archive format: {archive_path}")


def _ensure_download_if_missing(dataset_root: str, download_url: Optional[str]):
    """
    Downloads and extracts the dataset if the root directory is empty.
    This version handles archives with a single nested top-level directory to
    avoid paths like `CovidDataset/CovidDataset`.
    """
    if os.path.isdir(dataset_root) and os.listdir(dataset_root):
        print(f"[COVID] Dataset already exists at: {dataset_root}")
        return

    os.makedirs(dataset_root, exist_ok=True)

    if not download_url:
        print(f"[COVID] Created empty dataset root at: {dataset_root} (no download_url provided)")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "covid_dataset_archive.zip")
        extract_path = os.path.join(tmpdir, "extracted_data")

        print(f"[COVID] Downloading dataset from:\n  {download_url}")
        _download_file(download_url, archive_path)

        print(f"[COVID] Extracting dataset to a temporary location...")
        _extract_archive(archive_path, extract_path)

        # Check if the archive extracted into a single directory (e.g., 'CovidDataset')
        extracted_items = os.listdir(extract_path)
        source_dir = extract_path
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(extract_path, extracted_items[0])):
            # If so, set the source for the move operation to be that inner directory
            print(f"[COVID] Detected nested folder '{extracted_items[0]}', adjusting path.")
            source_dir = os.path.join(extract_path, extracted_items[0])

        # Move the actual content into the final dataset_root
        print(f"[COVID] Moving files to final destination: {dataset_root}")
        for item_name in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item_name)
            dest_item = os.path.join(dataset_root, item_name)
            shutil.move(source_item, dest_item)

    print(f"[COVID] Dataset successfully prepared at: {dataset_root}")


def _list_images_recursive(root_dir: str,
                           exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif")) -> List[
    str]:
    """Lists all image files recursively in a directory."""
    exts = tuple(e.lower() for e in exts)
    out = []
    for base, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(base, f))
    return sorted(out)


class _ListImageDataset(Dataset):
    """A dataset that loads images from a list of file paths."""

    def __init__(self, image_paths: List[str], transform=None, allow_empty: bool = False):
        if not image_paths and not allow_empty:
            raise RuntimeError("Empty image list for dataset.")
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ---------- main API ----------
def get_dataloader_covid19(
        dataset_root: str,
        input_size: int = 256,
        batch_size: int = 16,
        num_workers: int = 2,
        download_url: Optional[str] = "https://figshare.com/ndownloader/files/50920287",
        normal_keywords: Iterable[str] = ("normal", "no finding", "negative"),
        covid_keywords: Iterable[str] = ("Covid", "covid", "sars-cov", "sarscov", "pneumonia", "opacity", "infiltrate"),
):
    """
    Downloads and prepares the COVID-19 dataset and returns data loaders.
    - Downloads the dataset if `dataset_root` is missing or empty.
    - Splits the data into training, normal testing, and abnormal (COVID) testing sets.
    """
    # 0) Download and extract the dataset if it's missing
    _ensure_download_if_missing(dataset_root, download_url)

    # 1) Scan for all images within the dataset directory
    train_imgs = _list_images_recursive(os.path.join(dataset_root, 'Train/Normal'))
    test_normal_imgs = _list_images_recursive(os.path.join(dataset_root, 'Val/Normal'))
    test_abnormal_imgs = _list_images_recursive(os.path.join(dataset_root, 'Val/Covid'))
    if not train_imgs:
        raise RuntimeError(
            f"No images found in {os.path.join(dataset_root, 'Train/Normal')}. Please check the dataset integrity.")
    if not test_normal_imgs:
        raise RuntimeError(
            f"No images found in {os.path.join(dataset_root, 'Val/Normal')}. Please check the dataset integrity.")
    if not test_abnormal_imgs:
        raise RuntimeError(
            f"No images found in {os.path.join(dataset_root, 'Val/Covid')}. Please check the dataset integrity.")

    # 4) Define image transformations
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    train_tfm = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    test_tfm = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # 5) Create datasets and dataloaders
    train_dataset = _ListImageDataset(train_imgs, transform=train_tfm)
    test_normal_dataset = _ListImageDataset(test_normal_imgs, transform=test_tfm)
    test_abnormal_dataset = _ListImageDataset(test_abnormal_imgs, transform=test_tfm, allow_empty=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    test_normal_loader = DataLoader(test_normal_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    pin_memory=True)
    test_abnormal_loader = DataLoader(test_abnormal_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True)

    print("-" * 36)
    print(f"[COVID] Train (Normal): {len(train_dataset)} images")
    print(f"[COVID] Test  (Normal): {len(test_normal_dataset)} images")
    print(f"[COVID] Test  (Covid):  {len(test_abnormal_dataset)} images")
    if len(test_abnormal_dataset) == 0:
        print("[INFO] 'test_abnormal_loader' is empty as no COVID-19 images were found.")
    print("-" * 36)

    return train_loader, test_normal_loader, test_abnormal_loader
