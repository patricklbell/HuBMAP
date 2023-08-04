# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#BCE-Dice-Loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = alpha * BCE + (1 - alpha) * dice_loss
        
        return Dice_BCE

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    
FocalLoss_ALPHA = 0.95
FocalLoss_GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=FocalLoss_ALPHA, gamma=FocalLoss_GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    
TverskyLoss_ALPHA = 0.5
TverskyLoss_BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=TverskyLoss_ALPHA, beta=TverskyLoss_BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
    
FocalTverskyLoss_ALPHA = 0.5
FocalTverskyLoss_BETA = 0.5
FocalTverskyLoss_GAMMA = 1

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=FocalTverskyLoss_ALPHA, beta=FocalTverskyLoss_BETA, gamma=FocalTverskyLoss_GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
    
class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()
    
# Lovasz Helpers
def lovasz_hinge2(logits, labels, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    loss = torch.mean(lovasz_hinge_flat2(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                      for log, lab in zip(logits, labels))

    return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat2(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    weight = 1
    if labels.sum() == 0:
        weight = 0
    loss = torch.dot(F.relu(errors_sorted), Variable(grad)) * weight
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

class LovaszHingeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        # inputs = F.sigmoid(inputs)
        inputs = inputs.squeeze(1)
        targets = targets.squeeze(1)
        Lovasz = lovasz_hinge2(inputs, targets, per_image=False)                       
        return Lovasz

######################
# Evaluation Metrics #
######################
    
# https://arxiv.org/pdf/1602.06541.pdf
# Mean Pixel Accuracy for Binary Classification
class PixelAccuracy(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PixelAccuracy, self).__init__()

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = (F.sigmoid(inputs) > 0.5).int()       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = targets.size()
        
        return intersection / total

# https://arxiv.org/pdf/1602.06541.pdf
# F-Metric
FScore_BETA = 1.
class FScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FScore, self).__init__()

    def forward(self, inputs, targets, beta=FScore_BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = (F.sigmoid(inputs) > 0.5).int()
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1) > 0
        targets = targets.view(-1) > 0

        tp = (inputs & targets).sum()
        fp = (inputs & (~targets)).sum()
        fn = ((~inputs) & targets).sum()

        if not torch.is_nonzero(tp):
            return tp
        
        return (1+beta**2) * (tp / (tp*(1+beta**2) + fn + fp)) 

# https://arxiv.org/abs/2202.05273
class SensitivityScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SensitivityScore, self).__init__()

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = (F.sigmoid(inputs) > 0.5).int()
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1) > 0
        targets = targets.view(-1) > 0

        tp = (inputs & targets).sum()
        fn = ((~inputs) & targets).sum()

        if not torch.is_nonzero(tp):
            return tp
        
        return tp / (tp + fn) 

# https://arxiv.org/abs/2202.05273
class SpecificityScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SpecificityScore, self).__init__()

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = (F.sigmoid(inputs) > 0.5).int()
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1) > 0
        targets = targets.view(-1) > 0
        
        tn = ((~inputs) & (~targets)).sum()
        fp = (inputs & (~targets)).sum()

        if not torch.is_nonzero(tn):
            return tn
        
        return tn / (tn + fp) 