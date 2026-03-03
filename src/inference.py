"""
Inference script for skin lesion segmentation using a pretrained U-Net model

Usage:
    python src/inference.py \
        --image-path data/raw/ISIC2018_Task1-2_Test_Input \
        --model-weight models/best_model.pth \
        --output-path prediction
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import build_unet


# Preprocessing

def get_transform(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_image(image_path, transform):
    """Load a single image and return preprocessed tensor + original size."""
    img        = Image.open(image_path).convert("RGB")
    orig_size  = img.size             # (W, H) — saved for resizing mask back
    arr        = np.array(img)
    tensor     = transform(image=arr)["image"].unsqueeze(0)
    return tensor, orig_size


# Inference

@torch.no_grad()
def predict_batch(model, tensor, device, threshold=0.5):
    """Run model on a single preprocessed image tensor."""
    tensor = tensor.to(device)
    prob   = model(tensor).sigmoid().squeeze().cpu().numpy()
    mask   = (prob > threshold).astype(np.uint8) * 255
    return mask


# Main

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    image_dir  = Path(args.image_path)
    weight_path = Path(args.model_weight)
    output_dir  = Path(args.output_path)

    if not image_dir.exists():
        print(f"\nERROR: Image directory not found: {image_dir}")
        return

    if not weight_path.exists():
        print(f"\nERROR: Model weights not found: {weight_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    extensions  = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    ])

    if not image_paths:
        print(f"\nERROR: No images found in {image_dir}")
        print(f"Supported formats: {extensions}\n")
        return

    print(f"Images found : {len(image_paths)}")
    print(f"Weights      : {weight_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Threshold    : {args.threshold}\n")

    # --- Load model ---
    model = build_unet()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.\n")

    # --- Run inference ---
    transform  = get_transform(args.image_size)
    failed     = []

    for image_path in tqdm(image_paths, desc="Generating masks"):
        try:
            tensor, orig_size = load_image(image_path, transform)
            mask              = predict_batch(model, tensor, device, args.threshold)

            # Resize mask back to original image dimensions
            mask_img    = Image.fromarray(mask)
            mask_resized = mask_img.resize(orig_size, Image.NEAREST)

            # Save as PNG with matching filename
            out_name = image_path.stem + "_segmentation.png"
            mask_resized.save(output_dir / out_name)

        except Exception as e:
            failed.append((image_path.name, str(e)))

    # --- Summary ---
    total   = len(image_paths)
    success = total - len(failed)

    print(f"\n{'='*50}")
    print(f"  Done. {success}/{total} masks saved to: {output_dir}")
    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for name, err in failed:
            print(f"    {name}: {err}")
    print(f"{'='*50}\n")

    print("Output naming convention:")
    print("  Input : ISIC_XXXXXXX.jpg")
    print("  Output: ISIC_XXXXXXX_segmentation.png  (255=lesion, 0=background)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run skin lesion segmentation on a folder of images using a pretrained U-Net."
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to folder of input images (jpg/png)"
    )
    parser.add_argument(
        "--model-weight",
        required=True,
        help="Path to model weights file (.pt)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Folder to save output prediction masks"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size used during training (default: 256)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for binary mask (default: 0.5)"
    )
    args = parser.parse_args()
    main(args)