import os
import os.path as osp
import sys
import argparse
import subprocess


def main(args):
    sequences = [ osp.join(args['mot_dir'], seq) for seq in os.listdir(args['mot_dir']) ]
    for sequence_dir in sequences:
        cmdline = " ".join([
            sys.executable, "run.py",
            "--sequence", sequence_dir,
            # Dataset
            "--detector", str(args['detector']),
            "--min_obj_conf", str(args['min_obj_conf']),
            # Tracker
            "--tracker", str(args['tracker']),
            "--nms_iou_threshold", str(args['nms_iou_threshold']),
            # Track Management
            "--n_init", str(args['n_init']),
            "--n_lost", str(args['n_lost']),
            "--n_dead", str(args['n_dead']),
            "--pool_size", str(args['pool_size']),
            # Association
            "--degree", str(args['degree']),
            "--iou_dist_threshold", str(args['iou_dist_threshold']),
            "--cos_dist_threshold", str(args['cos_dist_threshold']),
            "--maha_iou_dist_threshold", str(args['maha_iou_dist_threshold']),
            "--maha_cos_dist_threshold", str(args['maha_cos_dist_threshold']),
            # Pseudo depth space setting
            "--max_depth", str(args['max_depth']),
            "--n_levels", str(args['n_levels']),
            # Runtime setting
            # =========================================================================
            "--output", str(args['output']),
            "--indoor" if args['indoor'] "",
            "--guess_limit", str(args['guess_limit']),
            ])
        proc = subprocess.Popen(cmdline, shell=True)
        proc.wait()
        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot_dir", required=True, help="MOT Sequence directory")
    # Dataset setting
    # =========================================================================
    parser.add_argument("--detector", default='default', type=str, help="which detector to use")
    parser.add_argument("--min_obj_conf", default=0.8, type=float, help="detected object confidence threshold")
    parser.add_argument("--indoor", action='store_true', help="indoor mode or outdoor mode")
    parser.add_argument("--guess_limit", default=1, help="number of guessing")
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
    parser.add_argument("--iou_dist_threshold", default=0.3, type=float, help="threshold for iou distance")
    parser.add_argument("--cos_dist_threshold", default=0.3, type=float, help="threshold for cos distance")
    parser.add_argument("--maha_iou_dist_threshold", default=0.5, type=float, help="threshold for maha iou distance")
    parser.add_argument("--maha_cos_dist_threshold", default=0.5, type=float, help="threshold for maha cos distance")
    # Pseudo depth space setting
    parser.add_argument("--max_depth", default=5, type=float, help="maximum depth range in meter")
    parser.add_argument("--n_levels", default=20, type=float, help="number of intervals between depth range")
    # Runtime setting
    # =========================================================================
    parser.add_argument("--output", default='output', help="output directoty")

    args = vars(parser.parse_args())
    main(args)
