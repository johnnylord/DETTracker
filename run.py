import os
import os.path as osp
import argparse
import pickle

import cv2
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm

from data.dataset.motsequence import MOTDSequence
from tracker.deepsort import DeepSORT
from tracker.deepsortplus import DeepSORTPlus
from utils.display import get_color, draw_box, draw_text
from utils.evaluation import export_results


def main(args):
    # Load MOT Sequence
    sequence = MOTDSequence(
                    root=args['sequence'],
                    detector=args['detector'],
                    min_conf_threshold=args['min_obj_conf'],
                    mode='test')
    print(sequence)

    # Create output directory
    output_dir = osp.join(args['output'], sequence.name)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # Load writer
    if args['export']:
        output = osp.join(output_dir, 'video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter(output, fourcc, sequence.fps,
                                (sequence.imgWidth, sequence.imgHeight))

    # Load Trackor
    if args['tracker'] == 'DeepSORT':
        tracker_cls = DeepSORT
    elif args['tracker'] == 'DeepSORTPlus':
        tracker_cls = DeepSORTPlus
    else:
        raise ValueError("Unknown tracker")

    tracker = tracker_cls(
                # Track FSM
                n_init=args['n_init'],
                n_lost=args['n_lost'],
                n_dead=args['n_dead'],
                # Association
                n_degree=args['degree'],
                iou_dist_threshold=args['iou_dist_threshold'],
                cos_dist_threshold=args['cos_dist_threshold'],
                # Pseudo depth
                n_levels=args['n_levels'],
                max_depth=args['max_depth'],
                pool_size=args['pool_size'],
                # Detection
                nms_iou_threshold=args['nms_iou_threshold'],
                )

    # Process video frame-by-frame
    results = {}
    inverse = T.ToPILImage()
    for idx in tqdm(range(len(sequence)), leave=True, desc="Processing"):
        frameId = idx + 1
        img, depthmap, flowmap, tboxes, bboxes, masks = sequence[idx]
        tracks = tracker(img, depthmap, flowmap, bboxes, masks)
        results[frameId] = tracks
        # Draw tracks (box + ID) on video frame
        if args['export']:
            frame = np.array(inverse(img))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for track in tracks:
                tid = track['id']
                box = track['box']
                text = f"ID:{tid}"
                color = get_color(tid)
                draw_box(frame, box, color=color)
                draw_text(frame, text, position=tuple(box[:2]), bgcolor=color)
            writer.write(frame)

    # Save tracking result
    export_results(results, output_dir, sequence.name)

    # Save tracking video
    if args['export']:
        writer.release()

    # Save tracking intermediate product
    with open(osp.join(output_dir, 'intermediate.pkl'), 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset setting
    # =========================================================================
    parser.add_argument("--sequence", required=True, help="mot sequence trackor will run on")
    parser.add_argument("--detector", default='default', type=str, help="which detector to use")
    parser.add_argument("--min_obj_conf", default=0.8, type=float, help="detected object confidence threshold")
    # Tracker setting
    # =========================================================================
    parser.add_argument("--tracker", default="DeepSORT", type=str, help="tracker to use")
    parser.add_argument("--nms_iou_threshold", default=1.0, type=float, help="maximum bbox overlapping")
    # Track Management
    parser.add_argument("--n_init", default=3, type=int, help="track activate threshold")
    parser.add_argument("--n_lost", default=3, type=int, help="track lost threshold")
    parser.add_argument("--n_dead", default=30, type=int, help="track dead threshold")
    parser.add_argument("--pool_size", default=100, type=int, help="reid feature pool set")
    # Assoication setting
    parser.add_argument("--degree", default=4, type=int, help="degree of freedom of kalman state")
    parser.add_argument("--iou_dist_threshold", default=0.3, type=float, help="gating threshold for iou distance")
    parser.add_argument("--cos_dist_threshold", default=0.3, type=float, help="gating threshold for cos distance")
    # Pseudo depth space setting
    parser.add_argument("--max_depth", default=5, type=float, help="maximum depth range in meter")
    parser.add_argument("--n_levels", default=30, type=float, help="number of intervals between depth range")
    # Runtime setting
    # =========================================================================
    parser.add_argument("--verbose", action='store_true', help="show information on terminal")
    parser.add_argument("--display", action='store_true', help="show processing result with opencv")
    parser.add_argument("--export", action='store_true', help="save processing result to process.mp4")
    parser.add_argument("--output", default='output', help="output directoty")

    args = vars(parser.parse_args())
    main(args)
