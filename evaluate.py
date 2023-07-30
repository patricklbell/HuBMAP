import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from utils.lovasz_loss import iou_binary

@torch.inference_mode()
def evaluate(model, dataloader, device, IoUThreshold=0.6):
    num_val_batches = len(dataloader)

    # @note Dice threshold of 0.6 given in HuBMAP evaluation section
    metric = MeanAveragePrecision(iou_type="segm", iou_thresholds=[IoUThreshold], compute_with_cache=False).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # iterate over the validation set and average the metric
    out = dict(mAP=0, BCE=0, IoU=0)
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        x, y_true = batch['image'], batch['mask']

        # move images and labels to correct device and type
        x = x.to(device=device, dtype=torch.float32)
        y_true = y_true.to(device=device, dtype=torch.long)

        # predict the mask
        y_pred = model(x)

        out['BCE'] += criterion(y_pred, y_true.float()).item()

        y_pred = (torch.sigmoid(y_pred) > 0.5).long()

        out['IoU'] += iou_binary(y_pred, y_true) / 100

        preds, targets = [], []
        scores = torch.tensor([1]).float()
        labels = torch.tensor([0]).int()
        for y_pred, y_true in zip(y_pred, y_true):
            preds.append(dict(masks=y_pred.unsqueeze(0) > 0, scores=scores, labels=labels))
            targets.append(dict(masks=y_true.unsqueeze(0) > 0, labels=labels))

        out['mAP'] += metric(preds, targets)['map'].item()

    out['BCE'] /= max(num_val_batches, 1)
    out['IoU'] /= max(num_val_batches, 1)
    out['mAP'] /= max(num_val_batches, 1)

    return out