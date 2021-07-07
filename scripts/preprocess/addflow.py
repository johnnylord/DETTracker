import os
import os.path as osp
import argparse
import math

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from model import PWCNet, transform
from utils import writeFlow


def main(args):
    floDir = osp.join(args['sequence'], 'flow')
    if not osp.exists(floDir):
        os.makedirs(floDir)
    # Aggregate image files
    imgDir = osp.join(args['sequence'], 'img1')
    imgFiles = sorted([ osp.join(imgDir, f) for f in os.listdir(imgDir) ])
    starts, stops = imgFiles[:-1], imgFiles[1:]
    starts.insert(0, starts[0])
    stops.insert(0, starts[0])

    # CUDA Environment
    device = torch.cuda.device(0)
    torch.cuda.set_device(device)

    # Load model
    model = PWCNet()
    model = model.cuda()
    model.eval()

    for idx, pairs in enumerate(zip(starts, stops)):
        # Raw Image
        img1 = Image.open(pairs[0])
        img2 = Image.open(pairs[1])
        # Preprocess image
        x1 = transform(img1)
        x2 = transform(img2)
        # Image metadata
        imgWidth = x1.shape[2]
        imgHeight = x1.shape[1]
        alignWidth = int(math.floor(math.ceil(imgWidth/64.0)*64.0))
        alignHeight = int(math.floor(math.ceil(imgHeight/64.0)*64.0))
        # Load Dataset
        x1 = x1.cuda().view(1, 3, imgHeight, imgWidth)
        x2 = x2.cuda().view(1, 3, imgHeight, imgWidth)
        # Align input dataset
        x1 = F.interpolate(input=x1, size=(alignHeight, alignWidth), mode='bilinear', align_corners=False)
        x2 = F.interpolate(input=x2, size=(alignHeight, alignWidth), mode='bilinear', align_corners=False)
        # Model Inference
        with torch.no_grad():
            flow = model(x1, x2)
        flow = F.interpolate(input=flow, size=(imgHeight, imgWidth), mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= float(imgWidth) / float(alignWidth)
        flow[:, 1, :, :] *= float(imgHeight) / float(alignHeight)
        flow = flow.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        # Export result
        bname = osp.basename(pairs[1])
        flowname = osp.join(floDir, "{}.flo".format(bname.split(".")[0]))
        writeFlow(flowname, flow)
        print(f"Processing ({idx}/{len(starts)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", help="MOT Sequence")
    args = vars(parser.parse_args())
    main(args)
