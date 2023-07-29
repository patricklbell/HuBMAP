import torch
import torch.nn.functional as F
from utils.lovasz_loss import lovasz_hinge
from tqdm import tqdm

@torch.inference_mode()
def evaluate(net, criterion, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)

    # iterate over the validation set
    loss = 0
    with torch.no_grad():
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
                x, y_true = batch['image'], batch['mask']

                # move images and labels to correct device and type
                x = x.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                y_true = y_true.to(device=device, dtype=torch.long)

                # predict the mask
                y_pred = net(x)

                y_pred = y_pred.squeeze(1)
                y_true = y_true.squeeze(1).float()

                loss += torch.mean(criterion(y_pred, y_true) + lovasz_hinge(y_pred, y_true))

    net.train()
    return loss / max(num_val_batches, 1)