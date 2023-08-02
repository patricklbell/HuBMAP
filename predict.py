import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from utils.data_loading import BasicDataset, load_image
from unet import UNet
from utils.utils import plot_img_and_mask
import matplotlib.pyplot as plt

@torch.inference_mode()
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                transform=True):
    img = BasicDataset.prepare(full_img, scale_factor, transform)
    img = img.to(device=device, dtype=torch.float32).unsqueeze(0)

    net.eval()
    output = net(img)
    
    output = F.interpolate(output, (full_img.shape[1], full_img.shape[0]), mode='bilinear')
    mask = torch.sigmoid(output)
    return mask.squeeze().cpu().numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--transform', action='store_false', default=True, help="Don't apply transforms to input")
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    out[mask > 0] = 1

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = load_image(filename, False)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device,
                           transform=args.transform)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')

            true_mask_filename = filename.replace('train', 'train_masks')
            print(true_mask_filename)
            if os.path.exists(true_mask_filename):
                true_mask = load_image(true_mask_filename, True)
                plot_img_and_mask(img, mask, true_mask)
            else:    
                plot_img_and_mask(img, mask)
