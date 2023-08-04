import argparse
import logging
import os
import random
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import HubmapDataset
from utils.metrics import DiceLoss, ComboLoss, IoULoss, LovaszHingeLoss

import matplotlib.pyplot as plt

dir_img = Path('./data/train/')
dir_mask = Path('./data/train_masks/')
dir_models = Path('./checkpoints/')

def save_model(model, wandb, epoch):
    Path(dir_models).mkdir(parents=True, exist_ok=True)
    filename = f'{dir_models}/model{wandb.run.name}_epoch{epoch}.pth'
    torch.save(model.state_dict(), filename)
    logging.info(f'Checkpoint {epoch} saved!')

def train_model(
        model,
        device,
        architecture: str,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        ignore: float = 0.0,
        es: bool = False,
        es_patience: int = 10,
        IoUThreshold=0.6,
        do_transform: bool = True,
        do_wandb: bool = True,
        do_save_model: bool = True
):
    # 1. Create dataset
    dataset = HubmapDataset(dir_img, dir_mask, scale=img_scale, do_transform=True)

    # 2. Ignore some data to speed up testing
    if ignore > 0.0:
        logging.warning(f'Ignoring {ignore*100}% of input dataset')
        dataset = Subset(dataset, random.sample(range(0, len(dataset)), int(len(dataset)*(1-ignore))))

    # 3. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 4. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    if do_wandb:
        experiment = wandb.init(project='HuBMAP')
        experiment.config.update(dict(
            architecture=architecture,
            epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            val_percent=val_percent, 
            save_checkpoint=save_checkpoint, 
            img_scale=img_scale, 
            es=es,
            es_patience=es_patience,
            IoUThreshold=IoUThreshold,
            do_transform=do_transform,
            loss='BCELoss',
        ))

    logging.info(f'''Starting training:
        Architecture:    {architecture}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        IoU Threshold:   {IoUThreshold}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.8, patience=5, min_lr=1e-5)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # 5. Begin training
    global_step = 0
    best_dsc = None
    best_epoch = 0
    es_counter = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        
        model.train()
        dataset.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                x, y_true = batch['image'], batch['mask']

                assert x.shape[1] == model.n_channels, f'''
                    Network has been defined with {model.n_channels} input channels, 
                    but loaded images have {x.shape[1]} channels. Please check that 
                    the images are loaded correctly.
                '''

                x = x.to(device=device, dtype=torch.float32)
                y_true = y_true.to(device=device, dtype=torch.float32)
                
                y_pred = model(x)

                loss = criterion(y_pred, y_true)
                
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()

                global_step += 1
                epoch_loss += loss.item()

                if do_wandb:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.update(x.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        dataset.eval()
        model.eval()
        eval = evaluate(model, val_loader, device, IoUThreshold=IoUThreshold)

        # scheduler.step(eval['DSC'])

        # early stopping
        if best_dsc is None or eval['DSC'] > best_dsc:
            if do_save_model:
                save_model(model, wandb, epoch)
            
            es_counter = 0
            best_epoch = epoch
            best_iou = eval['DSC']
        else:
            es_counter += 1
            if es and es_counter > es_patience:
                logging.info(f'Stopping run early because there were {es_counter} unsuccessful runs')
                if save_model:
                    save_model(model, wandb, epoch)
                break
        
        logging.info(f'''
        Validation DSC: {eval["DSC"]}
        Validation IoU: {eval["IoU"]}
        Validation Sensitivity: {eval["Sensitivity"]}
        Validation Specificity: {eval["Specificity"]}
        ''')
        if do_wandb:
            histograms = {}
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                if not (torch.isinf(value) | torch.isnan(value)).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'DSC':eval['DSC'],
                'IoU':eval['IoU'],
                'Sensitivity':eval['Sensitivity'],
                'Specificity':eval['Specificity'],
                'step': global_step,
                'epoch': epoch,
                'epoch loss': epoch_loss / len(train_loader),
                **histograms
            })
    
    if do_wandb and do_save_model:
        best_filename = f'{dir_models}/model{wandb.run.name}_epoch{best_epoch}.pth'
        artifact = wandb.Artifact(f'best-model', type='model')
        artifact.add_file(best_filename)
        experiment.log_artifact(artifact)
        experiment.finish()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--ignore', '-i', type=float, default=0, help='Proportion of examples to ignore, WARNING should only be used for testing')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_channels=1 for grayscale
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            architecture="UNet",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            ignore=args.ignore
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError!\n Enabling checkpointing to reduce memory usage, but this slows down training.')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            architecture="UNet",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            ignore=args.ignore
        )
