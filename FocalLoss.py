from torch import nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        # Use BCEWithLogitsLoss as base for numerical stability
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        # BCE loss
        bce_loss = self.bce(inputs, targets)
        
        # Apply sigmoid to input for probability
        probs = torch.sigmoid(inputs)
        
        # Calculate focal weight
        p_t = torch.where(targets == 1, probs, 1-probs)
        alpha_factor = torch.where(targets == 1, self.alpha, 1-self.alpha)
        focal_weight = alpha_factor * (1 - p_t) ** self.gamma
        
        # Apply weights and reduction
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss