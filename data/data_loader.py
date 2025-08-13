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
    DATA_DIR = "../data/dataset/chest_xray/chest_xray"
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


def _safe_loadmat(path: str):
    from scipy.io import loadmat
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        import h5py, numpy as np
        from types import SimpleNamespace

        with h5py.File(path, 'r') as f:
            if 'cjdata' not in f:
                raise RuntimeError(f"'cjdata' group not found in {path}")
            cj = f['cjdata']

            if isinstance(cj, h5py.Group):
                # read datasets directly
                image = cj['image'][()]            # (H,W) or (H,W,1)
                label = int(np.array(cj['label'][()]).squeeze())

                # optional fields
                pid = ''
                if 'PID' in cj:
                    pid_raw = cj['PID'][()]
                    if isinstance(pid_raw, (bytes, bytearray)):
                        pid = pid_raw.decode('utf-8', errors='ignore')
                    else:
                        arr = np.array(pid_raw).squeeze()
                        if arr.dtype.kind in {'u','i'}:
                            pid = ''.join(chr(int(x)) for x in arr.ravel().tolist())
                        else:
                            try:
                                pid = arr.tobytes().decode('utf-8', errors='ignore')
                            except Exception:
                                pid = str(arr)

                tumorMask = None
                if 'tumorMask' in cj:
                    tumorMask = cj['tumorMask'][()]

                return {'cjdata': SimpleNamespace(image=image, label=label, PID=pid, tumorMask=tumorMask)}

            else:
                raise RuntimeError(f"Unsupported 'cjdata' node type: {type(cj)}; expected h5py.Group")



class BrainTumor(Dataset):
    def __init__(self,
                 root_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 include_labels: Optional[List[int]] = None,
                 return_mask: bool = False):
        self.mat_paths = sorted(glob.glob(os.path.join(root_dir, '**', '*.mat'), recursive=True))
        if not self.mat_paths:
            raise RuntimeError(f"No .mat files found under {root_dir}")

        self.transform = transform
        self.include_labels = set(include_labels) if include_labels is not None else None
        self.return_mask = return_mask

        filtered = []
        self._meta = []
        for p in self.mat_paths:
            data = _safe_loadmat(p)
            cj = data['cjdata']
            label = int(cj.label)
            if self.include_labels is not None and label not in self.include_labels:
                continue
            filtered.append(p)
            pid = getattr(cj, 'PID', '') if hasattr(cj, 'PID') else ''
            self._meta.append((p, label, pid))
        self.mat_paths = filtered
        if not self.mat_paths:
            raise RuntimeError(f"No files remain after filtering by labels {self.include_labels}")

        if self.transform is None:
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std  = [0.229, 0.224, 0.225]
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ])

    def __len__(self):
        return len(self.mat_paths)

    def __getitem__(self, idx):
        path, label, pid = self._meta[idx]
        data = _safe_loadmat(path)
        cj = data['cjdata']

        img = np.array(cj.image)
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            img = img - img.min()
            denom = (img.max() - img.min()) if img.max() > img.min() else 1.0
            img = (img / denom * 255.0).astype(np.uint8)

        pil = Image.fromarray(img, mode='L').convert('RGB')
        image = self.transform(pil) if self.transform else pil

        if self.return_mask and hasattr(cj, 'tumorMask') and cj.tumorMask is not None:
            m = np.array(cj.tumorMask).astype(np.uint8)
            if m.ndim == 2:
                mask_t = torch.from_numpy(m[None, ...])  # [1,H,W]
            else:
                mask_t = torch.from_numpy(m.squeeze()[None, ...])
        else:
            mask_t = torch.zeros(1, image.shape[-2], image.shape[-1], dtype=torch.uint8)

        return (image, int(label), str(pid), mask_t)


def get_dataloaders_brain_tumor(
        dataset_root: str = r"dataset",
        input_size: int = 256,
        batch_size: int = 16,
        normal_label: int = 3,     # pick which tumor class to treat as "normal" (1,2,3)
        num_workers: int = 2,
        pin_memory: bool = True,
):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_ds = BrainTumor(
        root_dir=dataset_root,
        transform=train_transform,
        include_labels=[normal_label],
        return_mask=False
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    test_normal_ds = BrainTumor(
        root_dir=dataset_root,
        transform=test_transform,
        include_labels=[normal_label],
        return_mask=True
    )
    other_labels = [l for l in (1,2,3) if l != normal_label]
    test_abnormal_ds = BrainTumor(
        root_dir=dataset_root,
        transform=test_transform,
        include_labels=other_labels,
        return_mask=True
    )

    test_normal_loader   = DataLoader(test_normal_ds, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=pin_memory)
    test_abnormal_loader = DataLoader(test_abnormal_ds, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=pin_memory)

    print(f"[Figshare] Train (label={normal_label}) images: {len(train_ds)}")
    print(f"[Figshare] Test  normal (label={normal_label}) images: {len(test_normal_ds)}")
    print(f"[Figshare] Test  abnormal (labels={other_labels}) images: {len(test_abnormal_ds)}")

    return train_loader, test_normal_loader, test_abnormal_loader