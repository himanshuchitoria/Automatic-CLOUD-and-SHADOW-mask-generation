# src/models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation.
    Reduces the relative loss for well-classified examples, focusing on hard examples.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] (raw, unnormalized scores)
            targets: [B, H, W] (integer class labels)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] (raw, unnormalized scores)
            targets: [B, H, W] (integer class labels)
        """
        num_classes = logits.shape[1]
        logits = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)  # sum over batch, height, width
        intersection = torch.sum(logits * targets_one_hot, dims)
        union = torch.sum(logits + targets_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

def get_loss_function(name='FocalLoss'):
    """
    Utility function to select loss function by name.
    """
    if name.lower() == 'focalloss':
        return FocalLoss()
    elif name.lower() == 'diceloss':
        return DiceLoss()
    elif name.lower() == 'crossentropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")

# Example usage
if __name__ == '__main__':
    loss_fn = get_loss_function('FocalLoss')
    logits = torch.randn(2, 3, 256, 256)  # [batch, classes, H, W]
    targets = torch.randint(0, 3, (2, 256, 256))  # [batch, H, W]
    loss = loss_fn(logits, targets)
    print(f"Loss: {loss.item():.4f}")
