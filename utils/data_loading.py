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
from albumentations.pytorch import ToTensorV2


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)
    
def mean_and_std(filename):
    img = np.asarray(load_image(filename))
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
    # Basic
    RandomRotate90(p=1),
    HorizontalFlip(p=0.5),
    
    # Morphology
    GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
    GaussianBlur(blur_limit=(3,7), p=0.5),
    
    # Color
    RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, 
                                brightness_by_max=True,p=0.5),
    HueSaturationValue(hue_shift_limit=25, sat_shift_limit=30, 
                        val_shift_limit=0, p=0.5),
    
    
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
        w, h = image.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        assert image.size == mask.size, f'Image and mask should be the same size, but are {image.size} and {mask.size}'
        image = image.resize((newW, newH), resample=Image.BICUBIC)
        mask  =  mask.resize((newW, newH), resample=Image.NEAREST)
        image, mask = np.asarray(image), np.asarray(mask)

        if not self.do_transform:
            return NO_TRANSFORM(image=image/255, mask=mask/255)

        transform = TRAIN_TRANSFORM if self.training else EVAL_TRANSFORM
        return transform(image=image, mask=mask)
    
    @staticmethod
    def prepare(image, scale: float, do_transform: bool = True):
        w, h = image.size
        newW, newH = int(scale * w), int(scale * h)
        
        image = image.resize((newW, newH), resample=Image.BICUBIC)
        image = np.asarray(image)

        if not do_transform:
            return NO_TRANSFORM(image=image / 255)['image']
        
        return EVAL_TRANSFORM(image=image)['image']

    def __getitem__(self, idx):
        id = self.ids[idx]
        mask_file = list(self.mask_dir.glob(id + '.*'))
        img_file = list(self.images_dir.glob(id + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {id}: {img_file}'
        assert len(mask_file) == 1, f'Either no image or multiple images found for the ID {id}: {img_file}'
        img = load_image(img_file[0])
        mask = load_image(mask_file[0])

        out = self.preprocess(img, mask, self.scale)

        return {
            'image': out['image'].float(),
            'mask': out['mask'].long(),
        }


class HubmapDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, do_transform=True):
        super().__init__(images_dir, mask_dir, scale, do_transform)
