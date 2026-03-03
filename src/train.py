import argparse
import os
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_unet, BCEDiceLoss
from evaluate import dice_coefficient


# Config
DATA_DIR = Path("data/raw")
IMAGE_DIR = DATA_DIR / "ISIC2018_Task1-2_Training_Input"
MASK_DIR  = DATA_DIR / "ISIC2018_Task1_Training_GroundTruth"
CHECKPOINT_DIR = Path("models")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Training

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice = 0, 0

    for images, masks in tqdm(loader, desc="Val  ", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        dice = dice_coefficient(logits.sigmoid(), masks)

        total_loss += loss.item()
        total_dice += dice.item()

    return total_loss / len(loader), total_dice / len(loader)


# Main

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        IMAGE_DIR, MASK_DIR,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    model = build_unet().to(device)
    criterion = BCEDiceLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5, verbose=True)

    best_dice = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step(val_dice)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            print(f"New best model saved (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for skin lesion segmentation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    args = parser.parse_args()
    main(args)
