import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, num_classes=1):
    
    #Returns a U-Net model.

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # raw logits — we apply sigmoid in loss/eval
    )
    return model


# Combined BCE + Dice Loss

class BCEDiceLoss(nn.Module):

    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1 - bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = logits.sigmoid()
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
