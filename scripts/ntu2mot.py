import os
import os.path as osp
import argparse
import cv2
import numpy as np
import pickle
from collections import OrderedDict

def export_seqinfo(output_dir, seqinfo):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(osp.join(output_dir, 'seqinfo.ini'), 'w') as f:
        f.write('[Sequence]\n')
        for k, v in seqinfo.items():
            f.write("{}={}\n".format(k, v))

def export_images(root, cap):
    if not osp.exists(root):
        os.makedirs(root)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        cv2.imwrite(osp.join(root, f"{count:06}.jpg"), frame)

def export_dets(root, dets):
    if not osp.exists(root):
        os.makedirs(root)

    # Convert bounding box format from (tlbr) to (tlwh)
    results = {}
    for idx, people in enumerate(dets):
        idx += 1
        results[idx] = []
        for person in people:
            box = person['bbox']
            xmin = box[0] if box[0] > 0 else 0
            ymin = box[1] if box[1] > 0 else 0
            width, height = box[2]-box[0], box[3]-box[1]
            conf = box[4]
            results[idx].append((xmin, ymin, width, height, conf))

    # Export to det.txt file
    lines = []
    keys = sorted(list(results.keys()))
    for k in keys:
        dets = results[k]
        for det in dets:
            xmin, ymin = det[0], det[1]
            width, height = det[2], det[3]
            conf = det[4]
            line = f"{k},-1,{xmin:.2f},{ymin:.2f},{width:.2f},{height:.2f},{conf:.2f},-1,-1,-1"
            lines.append(line)

    with open(osp.join(root, 'det.txt'), 'w') as f:
        content = '\n'.join(lines)
        f.write(content)

def export_gts(root, gts):
    if not osp.exists(root):
        os.makedirs(root)

    lines = [ line for line in gts.split('\n') if len(line) > 0 ]

    results = {}
    for line in lines:
        fields = [ float(f) for f in line.split(',') ]
        frame_id = int(fields[0])+1
        track_id = int(fields[1])
        xmin = fields[2] if fields[2] > 0 else 0
        ymin = fields[3] if fields[3] > 0 else 0
        width = fields[4]
        height = fields[5]
        if track_id not in results:
            results[track_id] = []
        results[track_id].append((frame_id, xmin, ymin, width, height))

    lines = []
    keys = sorted(list(results.keys()))
    for key in keys:
        trajectory = results[key]
        for box in trajectory:
            frame_id = box[0]
            xmin, ymin = box[1], box[2]
            width, height = box[3], box[4]
            line = f"{frame_id},{key},{xmin:.2f},{ymin:.2f},{width:.2f},{height:.2f},1,1,1"
            lines.append(line)

    with open(osp.join(root, 'gt.txt'), 'w') as f:
        content = '\n'.join(lines)
        f.write(content)

def main(args):
    # Read input streams
    video = cv2.VideoCapture(osp.join(args['input'], 'video.mp4'))
    depth = cv2.VideoCapture(osp.join(args['input'], 'depth.mp4'))
    midas = cv2.VideoCapture(osp.join(args['input'], 'midas.mp4'))

    # Sanity Check
    video_size = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    depth_size = int(depth.get(cv2.CAP_PROP_FRAME_COUNT))
    midas_size = int(midas.get(cv2.CAP_PROP_FRAME_COUNT))

    assert video_size == depth_size == midas_size

    # Load detection
    with open(osp.join(args['input'], 'bodyposes.pkl'), 'rb') as f:
        dets = pickle.load(f)

    # Load groundtruth
    with open(osp.join(args['input'], 'gt.txt'), 'r') as f:
        gts = f.read()

    # Convert video to images in MOT Format
    export_images(osp.join(args['output'], 'img1'), video)
    export_images(osp.join(args['output'], 'depth'), depth)
    export_images(osp.join(args['output'], 'midas'), midas)
    export_dets(osp.join(args['output'], 'det'), dets)
    export_gts(osp.join(args['output'], 'gt'), gts)

    # Export seqinfo.ini
    seqinfo = OrderedDict({
        'name': osp.basename(args['input']),
        'imDir': 'img1',
        'depthDir': 'midas',
        'frameRate': int(video.get(cv2.CAP_PROP_FPS)),
        'seqLength': int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        'imWidth': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'imHeight': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'imExt': '.jpg',
    })
    export_seqinfo(args['output'], seqinfo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert ntu sequence to mot sequence")
    parser.add_argument("--input", help="ntu sequence directory")
    parser.add_argument("--output", help="converted sequence directory")
    args = vars(parser.parse_args())
    main(args)
