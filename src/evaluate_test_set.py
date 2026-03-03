import argparse, json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ISICDataset, get_val_transforms
from model import build_unet
from evaluate import dice_coefficient, iou_score, pixel_accuracy


def style_axis(ax):
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

def unnormalize(tensor, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
    return (tensor * std + mean).clamp(0, 1)


@torch.no_grad()
def evaluate_test_set(model, loader, device):
    model.eval()
    per_image, all_dice, all_iou, all_acc = [], [], [], []

    for images, masks in tqdm(loader, desc="Evaluating"):
        images, masks = images.to(device), masks.to(device)
        probs = model(images).sigmoid()
        for i in range(len(images)):
            d   = dice_coefficient(probs[i:i+1], masks[i:i+1]).item()
            iou = iou_score(probs[i:i+1], masks[i:i+1]).item()
            acc = pixel_accuracy(probs[i:i+1], masks[i:i+1]).item()
            per_image.append({"dice": d, "iou": iou, "accuracy": acc,
                               "lesion_pct": masks[i].mean().item() * 100})
            all_dice.append(d); all_iou.append(iou); all_acc.append(acc)

    return {
        "dice_mean":     float(np.mean(all_dice)),
        "dice_std":      float(np.std(all_dice)),
        "dice_median":   float(np.median(all_dice)),
        "iou_mean":      float(np.mean(all_iou)),
        "iou_std":       float(np.std(all_iou)),
        "accuracy_mean": float(np.mean(all_acc)),
        "n_images":      len(all_dice),
    }, per_image


def print_results(agg):
    print("\n" + "="*50)
    print("  TEST SET RESULTS")
    print("="*50)
    print(f"  Images            : {agg['n_images']}")
    print(f"  Dice (mean +/- std) : {agg['dice_mean']:.4f} +/- {agg['dice_std']:.4f}")
    print(f"  Dice (median)       : {agg['dice_median']:.4f}")
    print(f"  IoU  (mean +/- std) : {agg['iou_mean']:.4f} +/- {agg['iou_std']:.4f}")
    print(f"  Pixel Accuracy      : {agg['accuracy_mean']:.4f}")
    print("="*50)
    print(f"\n  README table:\n"
          f"  | Dice Coefficient | {agg['dice_mean']:.3f} |\n"
          f"  | IoU (Jaccard)    | {agg['iou_mean']:.3f} |\n"
          f"  | Pixel Accuracy   | {agg['accuracy_mean']:.3f} |")
    print("="*50 + "\n")


def plot_score_distributions(per_image, save_path):
    dices = [r["dice"] for r in per_image]
    ious  = [r["iou"]  for r in per_image]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#0d1117')
    fig.suptitle('Per-Image Score Distribution — Test Set', color='white',
                 fontsize=13, fontweight='bold')
    for ax, data, color, title in [
        (axes[0], dices, '#3fb950', 'Dice Coefficient'),
        (axes[1], ious,  '#58a6ff', 'IoU (Jaccard)'),
    ]:
        style_axis(ax)
        ax.hist(data, bins=20, color=color, edgecolor='#0d1117', alpha=0.85)
        ax.axvline(np.mean(data),   color='white',   linestyle='--', linewidth=1.5,
                   label=f'Mean   {np.mean(data):.3f}')
        ax.axvline(np.median(data), color='#f78166', linestyle=':',  linewidth=1.5,
                   label=f'Median {np.median(data):.3f}')
        ax.set_title(title, color='white', fontweight='bold')
        ax.set_xlabel('Score', color='#8b949e')
        ax.set_ylabel('Number of images', color='#8b949e')
        ax.set_xlim(0, 1)
        ax.legend(facecolor='#161b22', labelcolor='white', edgecolor='#30363d')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Saved -> {save_path}")
    plt.show()


def plot_best_and_worst(model, loader, device, per_image, save_path, n=3):
    all_imgs, all_msks = [], []
    for imgs, msks in loader:
        all_imgs.append(imgs); all_msks.append(msks)
    all_imgs = torch.cat(all_imgs).to(device)
    all_msks = torch.cat(all_msks).to(device)
    with torch.no_grad():
        all_probs = model(all_imgs).sigmoid()
    all_preds = (all_probs > 0.5).float()

    indexed  = [{"idx": i, **r} for i, r in enumerate(per_image)]
    sorted_r = sorted(indexed, key=lambda x: x["dice"])
    worst_n, best_n = sorted_r[:n], sorted_r[-n:][::-1]

    fig, axes = plt.subplots(n*2, 4, figsize=(16, n*2*3.5), facecolor='#0d1117')
    fig.suptitle('Best & Worst Predictions — Test Set', color='white',
                 fontsize=14, fontweight='bold', y=0.998)
    for j, t in enumerate(['Input', 'Ground Truth', 'Prediction', 'Error Map']):
        axes[0, j].set_title(t, color='#58a6ff', fontsize=11, fontweight='bold', pad=10)

    def draw(ax_row, idx, label, bg_color):
        img  = unnormalize(all_imgs[idx], device).permute(1,2,0).cpu().numpy()
        mask = all_msks[idx, 0].cpu().numpy()
        pred = all_preds[idx, 0].cpu().numpy()
        d, iou = per_image[idx]["dice"], per_image[idx]["iou"]
        emap = np.zeros((*mask.shape, 3))
        emap[(pred==1)&(mask==1)] = [0.18, 0.80, 0.44]
        emap[(pred==1)&(mask==0)] = [0.90, 0.25, 0.25]
        emap[(pred==0)&(mask==1)] = [0.25, 0.55, 0.95]
        for j, (data, cmap, vmin, vmax) in enumerate([
            (img, None, None, None), (mask, 'gray', 0, 1),
            (pred, 'gray', 0, 1),   (emap, None, None, None),
        ]):
            ax_row[j].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax_row[j].set_xticks([]); ax_row[j].set_yticks([])
            for sp in ax_row[j].spines.values(): sp.set_edgecolor('#30363d')
        ax_row[0].text(5, 20, f'{label}\nDice {d:.3f}  IoU {iou:.3f}',
            color='white', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, alpha=0.85))

    for i, r in enumerate(best_n):  draw(axes[i],   r["idx"], f'Best #{i+1}',  '#145a32')
    for i, r in enumerate(worst_n): draw(axes[n+i], r["idx"], f'Worst #{i+1}', '#7b241c')

    patches = [
        mpatches.Patch(color=[0.18,0.80,0.44], label='True Positive'),
        mpatches.Patch(color=[0.90,0.25,0.25], label='False Positive'),
        mpatches.Patch(color=[0.25,0.55,0.95], label='False Negative'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, facecolor='#0d1117',
               edgecolor='#30363d', labelcolor='white', fontsize=10,
               bbox_to_anchor=(0.5, 0.002))
    plt.tight_layout(rect=[0, 0.03, 1, 0.997])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Saved -> {save_path}")
    plt.show()


def main(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = Path(args.image_dir)
    mask_dir  = Path(args.mask_dir)

    if not image_dir.exists():
        print(f"\nERROR: Not found: {image_dir}")
        print("Download from https://challenge.isic-archive.com/data/#2018")
        print("  ISIC2018_Task1-2_Validation_Input")
        print("  ISIC2018_Task1_Validation_GroundTruth")
        print("Unzip both into data/raw/\n")
        return

    dataset = ISICDataset(image_dir, mask_dir, transform=get_val_transforms(args.image_size))
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    print(f"Test images: {len(dataset)}")

    model = build_unet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Checkpoint: {args.checkpoint}\n")

    agg, per_image = evaluate_test_set(model, loader, device)
    print_results(agg)

    Path("results").mkdir(exist_ok=True)
    with open("results/test_set_metrics.json", "w") as f:
        json.dump(agg, f, indent=2)
    print("Saved -> results/test_set_metrics.json")

    print("\nGenerating score distribution plot...")
    plot_score_distributions(per_image, "results/figures/test_score_distribution.png")

    print("Generating best & worst figure...")
    plot_best_and_worst(model, loader, device, per_image,
                         "results/figures/test_best_worst.png", n=args.n_examples)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="models/best_model.pth")
    parser.add_argument("--image_dir",   default="data/raw/ISIC2018_Task1-2_Validation_Input")
    parser.add_argument("--mask_dir",    default="data/raw/ISIC2018_Task1_Validation_GroundTruth")
    parser.add_argument("--image_size",  type=int, default=256)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--n_examples",  type=int, default=3)
    args = parser.parse_args()
    main(args)