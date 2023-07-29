import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from utils.utils import labels_to_masks


def write_mask(masks, id):
    cv2.imwrite(f'./data/train_masks/{id}.tif', masks[id])


if __name__ == '__main__':
    tile_meta_filename = "./data/tile_meta.csv"
    wsi_meta_filename = "./data/wsi_meta.csv"
    labels_filename = "./data/polygons.jsonl"

    try:
        tile_meta = pd.read_csv(tile_meta_filename)
    except FileNotFoundError:
        print(f"Couldn't find {tile_meta_filename}")
    try:
        labels = pd.read_json(labels_filename, lines=True)
    except:
        print(f"Couldn't find {labels_filename}")
    try:
        wsi_meta = pd.read_csv(wsi_meta_filename)
    except:
        print(f"Couldn't find {wsi_meta_filename}")
    labels.set_index(['id'], inplace=True)
    tile_meta.set_index(['id'], inplace=True)
    wsi_meta.set_index(['source_wsi'], inplace=True)

    # Create masks for only blood vessels
    print("Converting polygonal masks into images")
    masks = labels_to_masks(labels, include=['blood_vessel'], dtype=np.uint8)
    masks = masks['blood_vessel']

    # Remove tiles which don't have a label
    meta = tile_meta[tile_meta.index.isin(labels.index)]

    print("Writing mask images")
    with Pool() as p:
        unique = list(tqdm(
            p.imap(partial(write_mask, masks), meta.index),
            total=len(meta.index)
        ))
