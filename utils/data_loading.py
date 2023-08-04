import logging
import numpy as np
import torch
import pandas as pd
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from albumentations import (Compose, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
                            ShiftScaleRotate, ElasticTransform,
                            GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
                            RandomBrightnessContrast, HueSaturationValue, IAASharpen,
                            RandomGamma, RandomBrightness, RandomBrightnessContrast,
                            GaussianBlur,CLAHE,
                            Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
                            Normalize, OneOf, NoOp, Resize)
from albumentations.augmentations import ToFloat
from albumentations.pytorch import ToTensorV2
import cv2


def load_image(filename, mask):
    img = cv2.imread(str(filename))
    if mask:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = np.zeros_like(img)
        mask[img > 0] = 1
        return mask
    
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
    
def mean_and_std(filename):
    img = load_image(filename, False)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2, 0, 1))

    mean = np.zeros((img.shape[0]))
    std  = np.zeros((img.shape[0]))
    for i, channel in enumerate(img):
        mean[i] += np.mean(channel)
        std[i] += np.std(channel)

    return mean, std


# https://pytorch.org/vision/stable/models.html
# Pytorch expects this normalization for pretrained models
# which is based on ImageNet's (http://image-net.org/explore_popular.php)
# normalization, in this case our images are not natural so this is not
# a very accurate approximation if not using pretrained weights
# @note assumes three channels
# MEAN = (0.485, 0.456, 0.406)
# STD  = (0.229, 0.224, 0.225)

# Mean and std from dataset
MEAN = (0.630399254117647, 0.4166460305882353, 0.6861500178823529)
STD = (0.14574793035294117, 0.1963170202745098, 0.12342092792156863)

# Based on https://github.com/tikutikutiku/kaggle-hubmap/blob/main/src/05_train_with_pseudo_labels/transforms.py
TRAIN_TRANSFORM = Compose([
    # # Basic
    # RandomRotate90(p=1),
    # HorizontalFlip(p=0.5),
    
    # # Morphology
    # GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
    # GaussianBlur(blur_limit=(3,7), p=0.5),
    
    # # Color
    # RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, 
    #                             brightness_by_max=True,p=0.5),
    # HueSaturationValue(hue_shift_limit=25, sat_shift_limit=30, 
    #                     val_shift_limit=0, p=0.5),
    
    
    Normalize(mean=MEAN, std=STD),
    ToTensorV2(transpose_mask=True),
])

EVAL_TRANSFORM = Compose([
    Normalize(mean=MEAN, std=STD),
    ToTensorV2(transpose_mask=True),
])

NO_TRANSFORM = Compose([
    ToTensorV2(transpose_mask=True),
])

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, do_transform: bool = True):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.do_transform = do_transform

        self.ids = [splitext(file)[0] for file in listdir(mask_dir) if isfile(join(mask_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {mask_dir}, make sure you put your images there')
        
        # logging.info('Computing mean and std..')
        # avg_mean, avg_std = None, None
        # for id in tqdm(self.ids):
        #     mean, std = mean_and_std(list(self.images_dir.glob(id + '.*'))[0])
        #     avg_mean = mean if avg_mean is None else avg_mean + mean
        #     avg_std  = std  if avg_std  is None else avg_std  + std 

        # avg_mean /= len(self.ids)
        # avg_std  /= len(self.ids)
        # print(avg_mean, avg_std)

        logging.info(f'Created dataset with {len(self.ids)} examples')
        self.train()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __len__(self):
        return len(self.ids)

    def preprocess(self, image, mask, scale: float):
        w,h = int(scale * image.shape[1]), int(scale * image.shape[0])
        assert w > 0 and h > 0, 'Scale is too small, resized images would have no pixel'
        image = cv2.resize(image, (w,h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (w,h), interpolation=cv2.INTER_AREA)

        if not self.do_transform:
            return NO_TRANSFORM(image=image, mask=mask)

        transform = TRAIN_TRANSFORM if self.training else EVAL_TRANSFORM
        return transform(image=image, mask=mask)
    
    @staticmethod
    def prepare(image, scale: float, do_transform: bool = True):
        w,h = int(scale * image.shape[1]), int(scale * image.shape[0])
        assert w > 0 and h > 0, 'Scale is too small, resized images would have no pixel'
        image = cv2.resize(image, (w,h), interpolation=cv2.INTER_AREA)
        
        if not do_transform:
            return NO_TRANSFORM(image=image)['image'].clone().detach()
        return EVAL_TRANSFORM(image=image)['image'].clone().detach()

    def __getitem__(self, idx):
        id = self.ids[idx]
        mask_file = list(self.mask_dir.glob(id + '.*'))
        image_file = list(self.images_dir.glob(id + '.*'))

        assert len(image_file) == 1, f'Either no image or multiple images found for the ID {id}: {image_file}'
        assert len(mask_file) == 1, f'Either no image or multiple images found for the ID {id}: {mask_file}'
        image = load_image(image_file[0].absolute(), False)
        mask = load_image(mask_file[0].absolute(), True)

        assert image.shape[:2] == mask.shape[:2], f'Image and mask should be the same size, but are {image.shape[:2]} and {mask.shape[:2]}'

        out = self.preprocess(image, mask, self.scale)

        return {
            'image': out['image'].clone().detach(),
            'mask': out['mask'].clone().detach().unsqueeze(0),
        }


class HubmapDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, do_transform=True):
        super().__init__(images_dir, mask_dir, scale, do_transform)
