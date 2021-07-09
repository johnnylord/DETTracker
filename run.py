import os
import os.path as osp
import argparse

from data.dataset.motsequence import MOTDSequence
from utils.display import get_color, draw_box, draw_text


def main(args):
    # Load MOT Sequence
    sequence = MOTDSequence(root=args['sequence'], detector=args['detector'], mode='test')
    print(sequence)

    # Load Trackor
    tracker = DeepSORTPlus()

    # Process video frame-by-frame
    for idx in range(len(sequence)):
        frameId = idx + 1
        img, depthmap, flowmap, tboxes, bboxes = sequence[idx]
        tracks = tracker(img, depthmap, flow, bboxes)
        # Draw tracks (box + ID) on video frame
        for track in tracks:
            tid = track['id']
            box = track['bbox']
            text = f"ID:{tid}"
            color = get_color(tid)
            draw_box(img, box, color=color)
            draw_text(img, text, position=tuple(box[:2]), bgcolor=color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", help="mot sequence trackor will run on")
    parser.add_argument("--detector", default='default', help="which detector to use")
    parser.add_argument("--verbose", action='store_true', help="show information on terminal")
    parser.add_argument("--display", action='store_true', help="show processing result with opencv")
    parser.add_argument("--export", action='store_true', help="save processing result to process.mp4")
