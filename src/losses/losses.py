import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import cv2  

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
    
class Weighted_Loss(nn.Module):
    def __init__(self, sigma, const):
        super(Weighted_Loss, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.sigma = sigma
        self.const = const
    
    def true_probs(self, probs, mask):
        mask_oh = torch.nn.functional.one_hot(mask, 2).permute(2,0,1)
        res, _= (probs * mask_oh).max(0)
        return res.reshape(*mask.shape).float()

    def weighted_loss(self, logits, mask, sigma=3, const=200, eps=torch.finfo(torch.float32).eps):
        mask_np = np.array(mask.squeeze(), dtype=np.uint8)
        contours, _= cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        zeros = np.zeros_like(mask, dtype=np.uint8)
        W = cv2.drawContours(zeros, contours, -1, (1), 3)
        W = cv2.GaussianBlur(W, (0, 0), sigma, sigma, cv2.BORDER_DEFAULT)*const
        W = torch.tensor(W).float()
        probs = torch.nn.functional.softmax(logits, dim=0)
        probs_oh, mask = self.true_probs(probs, mask), mask.squeeze()
        epsilon = torch.ones_like(probs_oh)*eps
        loss = mask.float() * torch.log(probs_oh+epsilon) + (1-mask.float())*torch.log(1-probs_oh+epsilon)
        weighted_loss = -(W*loss).sum()
        return weighted_loss 

    def forward(self, logits_batch, targets_batch):
        logits_batch, targets_batch = logits_batch.clone().cpu(), targets_batch.clone().cpu()
        losses = [self.weighted_loss(logits, target) for logits, target in zip(logits_batch, targets_batch)]
        mean_loss = sum(losses)/len(losses)
        return mean_loss
    
    