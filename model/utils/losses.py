import torch
import torch.nn.functional as F
import torch.nn as nn

# For all loss functions, predictions and labels are assumed to be of size N, H, W

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth
    
    def __call__(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights from 0 to 1. If None, no class balancing is used.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        return loss.mean()

# A weighted combination of focal and dice loss
# factor is how much proportion of the loss the focal loss should take
class FocalDiceLoss(nn.Module):
    def __init__(self, factor=0.5, smooth=1, gamma=2, alpha=None):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.factor = factor
    
    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        focal_loss = self.focal(inputs, targets)

        return self.factor * focal_loss + (1 - self.factor) * dice_loss