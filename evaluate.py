import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torchmetrics.detection import MeanAveragePrecision
from utils.metrics import FScore, IoULoss, PixelAccuracy

@torch.inference_mode()
def evaluate(model, dataloader, device, IoUThreshold=0.6):
    num_val_batches = len(dataloader)

    # @note Dice threshold of 0.6 given in HuBMAP evaluation section
    mAP_metric           = MeanAveragePrecision(iou_type="segm", iou_thresholds=[IoUThreshold], compute_with_cache=False).to(device)
    PixelAccuracy_metric = PixelAccuracy().to(device)
    FScore_metric        = FScore().to(device)
    IoU_metric           = IoULoss().to(device)

    # iterate over the validation set and average the metric
    out = dict(mAP=0, PixelAccuracy=0, FScore=0, IoU=0)
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        x, y_true = batch['image'], batch['mask']

        # move images and labels to correct device and type
        x = x.to(device=device, dtype=torch.float32)
        y_true = y_true.to(device=device, dtype=torch.float32)

        # predict the mask
        y_pred = model(x)

        out['PixelAccuracy'] += PixelAccuracy_metric(y_pred, y_true).item()
        out['FScore']        += FScore_metric(y_pred, y_true).item()
        out['IoU']           += 1 - IoU_metric(y_pred, y_true).item()

        y_pred = torch.sigmoid(y_pred) > 0.5
        y_true = y_true > 0

        preds, targets = [], []
        scores = torch.tensor([1]).float()
        labels = torch.tensor([0]).int()
        for y_pred, y_true in zip(y_pred, y_true):
            preds.append(dict(masks=y_pred, scores=scores, labels=labels))
            targets.append(dict(masks=y_true, labels=labels))

        out['mAP'] += mAP_metric(preds, targets)['map'].item()

    out['PixelAccuracy'] /= max(num_val_batches, 1)
    out['FScore']        /= max(num_val_batches, 1)
    out['IoU']           /= max(num_val_batches, 1)
    out['mAP']           /= max(num_val_batches, 1)

    return out