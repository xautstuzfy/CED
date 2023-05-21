import sys
sys.path.append('core')
import argparse
import glob
import numpy as np
import torch
from PIL import Image
import imageio

from core.ced import CED
from core.utils import flow_viz
from core.utils.utils import InputPadder
import os

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, flow_dir):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    imageio.imwrite(os.path.join(flow_dir, '00003.png'), flo)


def normalize(x):
    return x / (x.max() - x.min())


def demo(args):

    model = torch.nn.DataParallel(CED(args))
    model.load_state_dict(torch.load(args.model))
    print(f"Loaded checkpoint at {args.model}")

    model = model.module
    model.to(DEVICE)
    model.eval()

    flow_dir = os.path.join(args.path, args.model_name)
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg')) + \
                 glob.glob(os.path.join(args.path, '*.ppm')) + \
                 glob.glob(os.path.join(args.path, '*.bmp')) + \
                 glob.glob(os.path.join(args.path, '*.jpeg'))

        images = sorted(images)

        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image('D:\\RRAFT\\demo-frames\\VBOF\\00003_img1.jpg')
            image2 = load_image('D:\\RRAFT\\demo-frames\\VBOF\\00003_img2.jpg')

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            viz(image1, flow_up, flow_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../CED/checkpoints/MIXed.pth',
                        help="restore checkpoint")
    parser.add_argument('--model_name', help="", default="MIXed")
    parser.add_argument('--path', default='path', help="dataset for evaluation")
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--small', default=False,action='store_true', help='use small model')
    args = parser.parse_args()

    demo(args)
