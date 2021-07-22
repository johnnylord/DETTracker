import os
import os.path as osp
import configparser
import argparse
import numpy as np
import pandas as pd
from prettytable import PrettyTable


def main(args):
    table = PrettyTable()
    table.field_names = [
            "Sequence", "FPS", "Resolution", "Length",
            "Tracks", "Tr./Fr.", "Boxes", "Box/Fr.",
            "Class", "Camera", "Viewpoint", "Conditions"
            ]
    # List out a number of sequences in the MOT directory
    sequences = [ osp.join(args['mot'], seq) for seq in os.listdir(args['mot']) ]
    for seq in sequences:
        seqpath = osp.join(seq, 'seqinfo.ini')
        parser = configparser.ConfigParser()
        parser.read(seqpath)

        # Metadata
        name = parser['Sequence']['name']
        fps = int(parser['Sequence']['framerate'])
        imgWidth = int(parser['Sequence']['imwidth'])
        imgHeight = int(parser['Sequence']['imheight'])
        seqLength = int(parser['Sequence']['seqlength'])

        # Track & Bboxes
        gtfile = osp.join(seq, 'gt', 'gt.txt')
        detfile = osp.join(seq, 'det', 'det-mrcnn-processed-mask.txt')
        # Read as DataFrame
        gt = pd.read_csv(gtfile, header=None)
        det = pd.read_csv(detfile, header=None)
        # Profiling
        n_tracks = len(gt.index)
        n_dets = len(det.index)
        track_per_frame = n_tracks / seqLength
        det_per_frame = n_dets / seqLength
        # Fixed conditions
        cls = "person"
        camera = "static"
        viewpoint = "medium"
        explain = ""
        for idx, field in enumerate(name.split(".")[0].split("_")):
            if idx == 0:
                explain += f"{field[0]} people"
                continue
            if idx == 1:
                if field[0] == 's':
                    explain += " with similiar appearance"
                elif field[0] == 'd':
                    explain += " and diverse appearance"
                continue
            if idx == 2:
                if field[0] == 'p':
                    explain += " move around\nin a predicted motion pattern"
                elif field[0] == 'u':
                    explain += " move around\nin a unpredicted motion pattern"
                continue
            if idx == 3:
                if field[0] == 'p':
                    explain += " with less pose variation."
                elif field[0] == 'u':
                    explain += " with large pose variation."
                continue
        condition = f"indoor. {explain}"

        table.add_row([
            name, fps, (imgWidth, imgHeight), seqLength,
            n_tracks, f"{track_per_frame:.1f}", n_dets, f"{det_per_frame:.1f}",
            cls, camera, viewpoint, condition
            ])
        print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot", required=True, help="mot directory")
    args = vars(parser.parse_args())
    main(args)
