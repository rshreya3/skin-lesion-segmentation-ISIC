"""
Expects data/raw/ to contain:
  ISIC2018_Task1-2_Training_Input/     (*.jpg images)
  ISIC2018_Task1_Training_GroundTruth/ (*_segmentation.png masks)
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Augmentation pipelines
def get_train_transforms(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# Dataset

class ISICDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        self.image_paths = sorted(self.image_dir.glob("*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"No .jpg images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_name = img_path.stem + "_segmentation.png"
        mask_path = self.mask_dir / mask_name

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # grayscale

        # Binarize mask (ISIC masks are 0 or 255)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)  # add channel dim → (1, H, W)

        return image, mask


# build train/val split


def get_dataloaders(
    image_dir,
    mask_dir,
    image_size=256,
    batch_size=8,
    val_split=0.15,
    num_workers=4,
    seed=42,
):
    from torch.utils.data import DataLoader

    full_dataset = ISICDataset(image_dir, mask_dir)
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val

    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # Apply separate transforms after split
    train_ds.dataset.transform = get_train_transforms(image_size)
    val_ds.dataset.transform = get_val_transforms(image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
