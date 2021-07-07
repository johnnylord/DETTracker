import os
import os.path as osp
import argparse

import cv2
import torch
import numpy as np

# Load pretrained model
model_type = "DPT_Large"
midas = torch.hub.load('intel-isl/MiDaS', model_type)
midas = midas.to("cuda")
midas = midas.eval()

# Load transformation function
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
midas_transform = midas_transforms.dpt_transform

def estimate_depth(img):
    input_batch = midas_transform(img).to("cuda")
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    # Convert output to grayscale image
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = (2**8)-1
    grayscale = max_val*(depth-depth_min)/(depth_max-depth_min)
    grayscale = np.stack([grayscale]*3, axis=-1)
    return grayscale.astype('uint8')

def main(args):
    # Load input images
    imgDir = osp.join(args['sequence'], 'img1')
    imgFiles = [ osp.join(imgDir, f) for f in os.listdir(imgDir) ]

    # Perform depth estimation on all images
    midasDir = osp.join(args['sequence'], 'midas')
    if not osp.exists(midasDir):
        os.makedirs(midasDir)

    for f in imgFiles:
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = estimate_depth(img)
        output = osp.join(midasDir, osp.basename(f))
        cv2.imwrite(output, depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", help="sequence directory in MOT dataset")
    args = vars(parser.parse_args())
    main(args)
