import os
import os.path as osp
import argparse

import numpy as np
import torchvision.transforms as T

from data.dataset.motsequence import MOTDSequence
from tracker.deepsort import DeepSORT
from utils.display import get_color, draw_box, draw_text


def main(args):
    # Load MOT Sequence
    sequence = MOTDSequence(
                    root=args['sequence'],
                    detector=args['detector'],
                    min_conf_threshold=args['min_obj_conf'],
                    mode='test')
    print(sequence)

    # Load Trackor
    tracker = DeepSORT(
                n_init=args['n_init'],
                n_lost=args['n_lost'],
                n_dead=args['n_dead'],
                pool_size=args['pool_size'],
                iou_dist_threhsold=args['iou_dist_threhsold'],
                cos_dist_threhsold=args['cos_dist_threhsold'])
    print(tracker)

    # Process video frame-by-frame
    inverse = T.ToPILImage()
    for idx in range(len(sequence)):
        frameId = idx + 1
        img, depthmap, flowmap, tboxes, bboxes = sequence[idx]
        tracks = tracker(img, depthmap, flowmap, bboxes)
        # Draw tracks (box + ID) on video frame
        frame = np.array(inverse(img))
        for track in tracks:
            tid = track['id']
            box = track['box']
            text = f"ID:{tid}"
            color = get_color(tid)
            draw_box(frame, box, color=color)
            draw_text(frame, text, position=tuple(box[:2]), bgcolor=color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset setting
    # =========================================================================
    parser.add_argument("--sequence", required=True, help="mot sequence trackor will run on")
    parser.add_argument("--detector", default='default', type=str, help="which detector to use")
    parser.add_argument("--min_obj_conf", default=0.8, type=float, help="detected object confidence threshold")
    # Tracker setting
    # =========================================================================
    # Track Management
    parser.add_argument("--n_init", default=3, type=int, help="track activate threshold")
    parser.add_argument("--n_lost", default=3, type=int, help="track lost threshold")
    parser.add_argument("--n_dead", default=30, type=int, help="track dead threhsold")
    parser.add_argument("--pool_size", default=100, type=int, help="reid feature pool set")
    # Assoication setting
    parser.add_argument("--iou_dist_threhsold", default=0.3, type=float, help="gating threshold for iou distance")
    parser.add_argument("--cos_dist_threhsold", default=0.3, type=float, help="gating threshold for cos distance")
    # Runtime setting
    # =========================================================================
    parser.add_argument("--verbose", action='store_true', help="show information on terminal")
    parser.add_argument("--display", action='store_true', help="show processing result with opencv")
    parser.add_argument("--export", action='store_true', help="save processing result to process.mp4")

    args = vars(parser.parse_args())
    main(args)
