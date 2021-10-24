import os
import os.path as osp
import sys
sys.path.insert(0, osp.dirname(osp.dirname(__file__)))
import argparse
import pickle

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from data.dataset.motsequence import MOTDSequence
from utils.flow import flow2img, torchflow2img
from utils.display import draw_box, draw_mask, draw_text, get_color, get_color_mask


def main(args):
    if torch.cuda.is_available():
        torch.rand(1).cuda()

    sequence = MOTDSequence(root=args['sequence'],
                            detector=args['detector'],
                            mode='test')

    with open(args['result'], 'rb') as f:
        results = pickle.load(f)

    if not args['silent']:
        cv2.namedWindow('GT', cv2.WINDOW_GUI_EXPANDED)
        cv2.namedWindow('DET', cv2.WINDOW_GUI_EXPANDED)
        cv2.namedWindow('Depth', cv2.WINDOW_GUI_EXPANDED)
        cv2.namedWindow('Optical', cv2.WINDOW_GUI_EXPANDED)

    if args['export']:
        name = osp.basename(args['sequence'])
        if not osp.exists(name):
            os.makedirs(name)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        gt_writer = cv2.VideoWriter(f'{name}/gt.mp4', fourcc, sequence.fps, (sequence.imgWidth, sequence.imgHeight))
        det_writer = cv2.VideoWriter(f'{name}/det.mp4', fourcc, sequence.fps, (sequence.imgWidth, sequence.imgHeight))
        map_writer = cv2.VideoWriter(f'{name}/map.mp4', fourcc, sequence.fps, (sequence.imgWidth, sequence.imgHeight))
        flo_writer = cv2.VideoWriter(f'{name}/flo.mp4', fourcc, sequence.fps, (sequence.imgWidth, sequence.imgHeight))

    inverse = T.ToPILImage()
    for i in range(len(sequence)):
        print(f"Process {i}/{len(sequence)}", end='\r\b')
        img, depthmap, flow, tboxes, bboxes, masks = sequence[i]

        # Convert images to numpy frames
        img = np.array(inverse(img))
        depthmap = np.array(inverse(depthmap))
        if torch.cuda.is_available():
            flow = flow.cuda()
            flow = torchflow2img(flow)
        else:
            flow = flow2img(flow)

        # Convert RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gt_img = img.copy()
        det_img = img.copy()
        flow = cv2.cvtColor(flow, cv2.COLOR_RGB2BGR)
        depthmap = cv2.cvtColor(depthmap, cv2.COLOR_RGB2BGR)

        # Draw track on image
        frameId = i + 1
        tracks = results[frameId]
        for track in tracks:
            tid = track['id']
            text = f"ID:{tid}"
            color = get_color(tid)
            box = track['box']
            draw_box(gt_img, box, color=color)
            draw_text(gt_img, text, box[:2], bgcolor=color)

        # Draw bounding box and mask on image
        if 'mask' in args['detector']:
            for box, mask in zip(bboxes, masks):
                tid = int(box[0])
                color = get_color(tid)
                color_mask = get_color_mask(mask)
                tlwh = box[1:1+4]
                tlbr = [ tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3] ]
                draw_box(det_img, tlbr, color=color)
                draw_mask(det_img, tlbr, color_mask)
        else:
            for box in bboxes:
                tid = int(box[0])
                color = get_color(tid)
                tlwh = box[1:1+4]
                tlbr = [ tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3] ]
                draw_box(det_img, tlbr, color=color)

        if not args['silent']:
            cv2.imshow('GT', gt_img)
            cv2.imshow('DET', det_img)
            cv2.imshow('Optical', flow)
            cv2.imshow('Depth', depthmap)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        if args['export']:
            gt_writer.write(gt_img)
            det_writer.write(det_img)
            flo_writer.write(flow)
            map_writer.write(depthmap)

    if args['export']:
        gt_writer.release()
        det_writer.release()
        flo_writer.release()
        map_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, help="sequence directory")
    parser.add_argument("--result", required=True, help="intermediate result pickle file")
    parser.add_argument("--detector",
            default='mrcnn-processed-mask',
            choices=[
                'default-processed',
                'mrcnn-processed-mask',
                ],
            help="default detector")
    parser.add_argument("--silent", action='store_true')
    parser.add_argument("--export", action='store_true')
    args = vars(parser.parse_args())
    main(args)
