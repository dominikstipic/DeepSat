import torch
import torch.nn as nn
import torch.nn.functional as func

class Focal_Loss(nn.Module):
    types = ["exp", "poly"]
    reductions = ["mean", "sum", "none"]

    def __init__(self, gamma, type, reduction="mean"):
        super(Focal_Loss, self).__init__()
        assert type in self.types, "unknown focal loss type"
        assert reduction in self.reductions, "unknown reduction"
        self.gamma = gamma
        self.type = type
        self.reduction = reduction
    
    def exponential(self, pt, gamma):
        return torch.exp(-gamma*(1-pt))
    
    def polynomial(self, pt, gamma):
        return (1-pt).pow(gamma)

    def forward(self, logits, targets):
        b,c,h,w = logits.shape
        cross_entropy = func.cross_entropy(logits, targets, reduction="none").view(-1)
        logits = logits.view(b,c,h*w).permute(0,2,1).view(-1,c).contiguous()
        probs = torch.nn.functional.softmax(logits, dim=1)
        pt = probs.gather(1, targets.view(-1,1)).view(-1)
        if self.type == "poly":
            loss = self.polynomial(pt, self.gamma)
        elif self.type == "exp":    
            loss = self.exponential(pt, self.gamma)
        loss *= cross_entropy
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss
        return loss
    
    

