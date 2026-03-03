import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataset import get_dataloaders
from model import build_unet
from evaluate import dice_coefficient, iou_score


def unnormalize(tensor, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    return (tensor * std + mean).clamp(0, 1)

def style_axis(ax):
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')


def generate_prediction_grid(model, loader, device, save_path, n_rows=4):
    model.eval()
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        probs = model(images).sigmoid()
    preds = (probs > 0.5).float()

    fig, axes = plt.subplots(n_rows, 4, figsize=(16, n_rows * 4), facecolor='#0d1117')
    fig.suptitle('U-Net Skin Lesion Segmentation ISIC 2018 Validation Set',
                 color='white', fontsize=15, fontweight='bold', y=0.99)

    for j, title in enumerate(['Input Image', 'Ground Truth', 'Prediction', 'Error Map']):
        axes[0, j].set_title(title, color='#58a6ff', fontsize=12, fontweight='bold', pad=12)

    for i in range(n_rows):
        img  = unnormalize(images[i], device).permute(1, 2, 0).cpu().numpy()
        mask = masks[i, 0].cpu().numpy()
        pred = preds[i, 0].cpu().numpy()
        dice = dice_coefficient(probs[i:i+1], masks[i:i+1]).item()
        iou  = iou_score(probs[i:i+1], masks[i:i+1]).item()

        error_map = np.zeros((*mask.shape, 3))
        error_map[(pred==1) & (mask==1)] = [0.18, 0.80, 0.44]  # TP green
        error_map[(pred==1) & (mask==0)] = [0.90, 0.25, 0.25]  # FP red
        error_map[(pred==0) & (mask==1)] = [0.25, 0.55, 0.95]  # FN blue

        for j, (data, cmap, vmin, vmax) in enumerate([
            (img, None, None, None), (mask, 'gray', 0, 1),
            (pred, 'gray', 0, 1),   (error_map, None, None, None),
        ]):
            ax = axes[i, j]
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('#30363d')

        axes[i, 0].text(6, 20, f'Dice {dice:.3f}   IoU {iou:.3f}',
            color='white', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', alpha=0.85))

    legend_patches = [
        mpatches.Patch(color=[0.18, 0.80, 0.44], label='True Positive  (correct lesion)'),
        mpatches.Patch(color=[0.90, 0.25, 0.25], label='False Positive (over-segmented)'),
        mpatches.Patch(color=[0.25, 0.55, 0.95], label='False Negative (missed lesion)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               facecolor='#0d1117', edgecolor='#30363d',
               labelcolor='white', fontsize=11, bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Saved -> {save_path}")
    plt.show()


def generate_training_curves(train_losses, val_losses, val_dices, save_path):
    epochs     = range(1, len(train_losses) + 1)
    best_epoch = int(np.argmax(val_dices)) + 1
    best_dice  = max(val_dices)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
    fig.suptitle('Training History', color='white', fontsize=14, fontweight='bold')

    style_axis(ax1)
    ax1.plot(epochs, train_losses, color='#58a6ff', linewidth=2, label='Train Loss')
    ax1.plot(epochs, val_losses,   color='#f78166', linewidth=2, label='Val Loss')
    ax1.axvline(best_epoch, color='#ffffff', linestyle=':', alpha=0.4,
                label=f'Best epoch ({best_epoch})')
    ax1.set_title('Loss (BCE + Dice)', color='white', fontweight='bold')
    ax1.set_xlabel('Epoch', color='#8b949e'); ax1.set_ylabel('Loss', color='#8b949e')
    ax1.legend(facecolor='#161b22', labelcolor='white', edgecolor='#30363d')

    style_axis(ax2)
    ax2.plot(epochs, val_dices, color='#3fb950', linewidth=2)
    ax2.axhline(best_dice, color='#3fb950', linestyle='--', alpha=0.4,
                label=f'Best: {best_dice:.4f} (epoch {best_epoch})')
    ax2.scatter([best_epoch], [best_dice], color='#3fb950', s=80, zorder=5)
    ax2.set_title('Validation Dice Coefficient', color='white', fontweight='bold')
    ax2.set_xlabel('Epoch', color='#8b949e'); ax2.set_ylabel('Dice Score', color='#8b949e')
    ax2.set_ylim(0, 1)
    ax2.legend(facecolor='#161b22', labelcolor='white', edgecolor='#30363d')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Saved to {save_path}")
    plt.show()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_DIR = Path("data/raw/ISIC2018_Task1-2_Validation_Input")
    MASK_DIR  = Path("data/raw/ISIC2018_Task1_Validation_GroundTruth")
    _, val_loader = get_dataloaders(IMAGE_DIR, MASK_DIR, batch_size=8)

    model = build_unet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded: {args.checkpoint}")

    print("Generating prediction grid...")
    generate_prediction_grid(model, val_loader, device,
                              save_path="results/figures/results_grid.png",
                              n_rows=args.n_rows)

    history_path = Path("results/training_history.npy")
    if history_path.exists():
        print("Generating training curves...")
        history = np.load(history_path, allow_pickle=True).item()
        generate_training_curves(history["train_losses"], history["val_losses"],
                                  history["val_dices"],
                                  save_path="results/figures/training_curves.png")
    else:
        print("No training_history.npy found, skipping curves")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    parser.add_argument("--n_rows", type=int, default=4)
    args = parser.parse_args()
    main(args)