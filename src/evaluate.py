import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloaders, get_val_transforms
from model import build_unet

#metrics

def dice_coefficient(preds, targets, threshold=0.5, smooth=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_score(preds, targets, threshold=0.5, smooth=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def pixel_accuracy(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    correct = (preds == targets).float().sum()
    total = torch.numel(targets)
    return correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metrics = {"dice": [], "iou": [], "accuracy": []}

    for images, masks in tqdm(loader, desc="Evaluating"):
        images, masks = images.to(device), masks.to(device)
        probs = model(images).sigmoid()

        metrics["dice"].append(dice_coefficient(probs, masks).item())
        metrics["iou"].append(iou_score(probs, masks).item())
        metrics["accuracy"].append(pixel_accuracy(probs, masks).item())

    return {k: np.mean(v) for k, v in metrics.items()}




def visualize_predictions(model, loader, device, n=4, save_path="results/figures/sample_prediction.png"):
    model.eval()
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        probs = model(images).sigmoid()
        preds = (probs > 0.5).float()

    # Unnormalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    fig.suptitle("Skin Lesion Segmentation U-Net Predictions", fontsize=14, fontweight="bold")

    for i in range(n):
        img = (images[i] * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        mask = masks[i, 0].cpu().numpy()
        pred = preds[i, 0].cpu().numpy()

        axes[i, 0].imshow(img);         axes[i, 0].set_title("Input Image")
        axes[i, 1].imshow(mask, cmap="gray"); axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred, cmap="gray"); axes[i, 2].set_title("Prediction")

        for ax in axes[i]: ax.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization → {save_path}")
    plt.show()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR  = Path("data/raw")
    IMAGE_DIR = DATA_DIR / "ISIC2018_Task1-2_Validation_Input"
    MASK_DIR  = DATA_DIR / "ISIC2018_Task1_Validation_GroundTruth"

    _, val_loader = get_dataloaders(IMAGE_DIR, MASK_DIR, batch_size=args.batch_size)

    model = build_unet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    results = evaluate(model, val_loader, device)
    print("\n Evaluation Results ")
    print(f"  Dice Coefficient : {results['dice']:.4f}")
    print(f"  IoU (Jaccard)    : {results['iou']:.4f}")
    print(f"  Pixel Accuracy   : {results['accuracy']:.4f}")


    visualize_predictions(model, val_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args)
