import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics import FScore, SensitivityScore, SpecificityScore


@torch.inference_mode()
def evaluate(model, dataloader, device, IoUThreshold=0.6):
    num_val_batches = len(dataloader)

    FScore_metric = FScore().to(device)
    Sensitivity_metric = SensitivityScore().to(device)
    Specificity_metric = SpecificityScore().to(device)

    # iterate over the validation set and average the metric
    out = dict(DSC=0, IoU=0, Sensitivity=0, Specificity=0)
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        x, y_true = batch['image'], batch['mask']

        # move images and labels to correct device and type
        x = x.to(device=device, dtype=torch.float32)
        y_true = y_true.to(device=device, dtype=torch.float32)

        # predict the mask
        y_pred = model(x)

        out['DSC'] += FScore_metric(y_pred, y_true, 1).item()
        out['IoU'] += FScore_metric(y_pred, y_true, 0).item()
        out['Sensitivity'] += Sensitivity_metric(y_pred, y_true).item()
        out['Specificity'] += Specificity_metric(y_pred, y_true).item()

    out['DSC'] /= max(num_val_batches, 1)
    out['IoU'] /= max(num_val_batches, 1)
    out['Sensitivity'] /= max(num_val_batches, 1)
    out['Specificity'] /= max(num_val_batches, 1)

    return out