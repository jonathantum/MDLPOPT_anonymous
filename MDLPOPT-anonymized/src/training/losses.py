import torch
import torch.nn.functional as F
import torch.nn as nn

def masked_mse(pred, target, mask):
    # diff = (pred - target) ** 2
    # diff = diff * mask
    # return diff.sum() / mask.sum()

    loss = (((pred - target) ** 2) * mask).sum() / mask.sum()
    return loss

def weighted_masked_mae(pred, target, mask):
    # percentiles computed on-the-fly
    q50 = torch.quantile(target[mask.bool()], 0.50)
    q80 = torch.quantile(target[mask.bool()], 0.80)
    q95 = torch.quantile(target[mask.bool()], 0.95)

    weights = torch.ones_like(target)
    weights[target >= q50] = 1.5
    weights[target >= q80] = 3.0
    weights[target >= q95] = 5.0

    loss = torch.abs(pred - target) * mask * weights
    return loss.sum() / (mask * weights).sum()


def masked_mae(pred, target, mask):
    loss = torch.abs(pred - target) * mask
    return loss.sum() / mask.sum()

def weighted_masked_huber(pred, target, mask, delta=1.0):
    error = pred - target
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.tensor(delta, device=pred.device))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    # apply mask
    loss = loss * mask
    return loss.sum() / mask.sum()

def masked_huber(pred, target, mask, delta=1.0):
    err = pred - target
    abs_err = torch.abs(err)
    quad = torch.minimum(abs_err, torch.tensor(delta, device=pred.device))
    lin = abs_err - quad
    loss = 0.5 * quad**2 + delta * lin
    loss = loss * mask
    return loss.sum() / mask.sum()

def masked_quantile_loss(pred, target, mask, q=0.7):
    err = target - pred
    loss = torch.maximum(q * err, (q - 1) * err)
    loss = loss * mask
    return loss.sum() / mask.sum()

def hybrid_loss(pred, target, mask):
    mse = ((pred - target)**2) * mask

    tail_mask = (target >= 4).float()
    tail_loss = torch.abs(pred - target) * mask * tail_mask

    return mse.sum() / mask.sum() + 0.5 * tail_loss.sum() / (tail_mask * mask).sum()



import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, smoothing=0.1, reduction='mean'):
        super(FocalLoss, self).__init__()
        # alpha should be a tensor of shape [C] if provided
        self.alpha = alpha 
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = log_probs.exp()

        # 1. Label Smoothing
        with torch.no_grad():
            soft_targets = torch.full_like(inputs, self.smoothing / (num_classes - 1))
            soft_targets.scatter_(1, targets.view(-1, 1), 1.0 - self.smoothing)

        # 2. Focal Term: (1 - p)^gamma
        focal_weight = torch.pow(1 - probs, self.gamma)
        
        # 3. Combine: -y * (1-p)^gamma * log(p)
        loss = -soft_targets * focal_weight * log_probs
        
        # 4. Apply Alpha (Class Weights)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Gather weights for the target class for each sample
            at = self.alpha.gather(0, targets) # [N]
            loss = loss * at.view(-1, 1)

        loss = loss.sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()