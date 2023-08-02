import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import (polygon)
import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import pandas as pd
from functools import partial
from tqdm import tqdm

def plot_img_and_mask(img, mask, true_mask=None):
    img, mask = np.asarray(img).transpose((1,0,2)), np.asarray(mask).transpose((1,0))

    fig, ax = plt.subplots(1, 2 if true_mask is None else 3)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title(f'Mask')
    ax[1].imshow(mask)
    if true_mask is not None:
        true_mask = np.asarray(true_mask).transpose((1,0))
        ax[2].set_title(f'True Mask')
        ax[2].imshow(true_mask)
        
    plt.xticks([]), plt.yticks([])
    plt.show()

def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str


def coordinates_to_mask(coordinates: np.ndarray, w=512, h=512, dtype=np.float64) -> np.ndarray:
    img = np.zeros((w, h), dtype=dtype)

    for coords in coordinates:
        rr, cc = polygon(coords[:,1], coords[:,0], img.shape)
        img[rr,cc]=255

    return img


def mask_to_image(mask, rgba):
    bw = np.stack((mask,)*4, axis=-1)
    return bw * rgba


def label_to_mask(label, include=[], exclude=[], dtype=np.float64):
    coordinates = {}
    for annotation in label['annotations']:
        type = annotation['type']
        if type not in include or type in exclude:
            continue
        coordinates[type] = (coordinates[type] if type in coordinates else []) + [np.array(x) for x in annotation['coordinates']]

    for type in include:
        label[type] = coordinates_to_mask(coordinates[type] if type in coordinates else [], dtype=dtype)

    return label

def labels_to_masks(df: pd.DataFrame, include=[], exclude=[], dtype=np.float64) -> pd.DataFrame:
    tqdm.pandas()
    return df.progress_apply(partial(label_to_mask, include=include, exclude=exclude, dtype=dtype), axis=1)
