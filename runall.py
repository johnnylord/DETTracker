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
            "--detector", args['detector'],
            "--tracker", args['tracker'],
            "--degree", args['degree'],
            ])
        proc = subprocess.Popen(cmdline, shell=True)
        proc.wait()
        proc.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot_dir", required=True, help="MOT Sequence directory")
    parser.add_argument("--tracker", default='DeepSORT', type=str, help="which tracker to use")
    parser.add_argument("--detector", default='mrcnn-processed-mask', type=str, help="which detector to use")
    parser.add_argument("--degree", default=3, type=str, help="degree of freedom of kalman state")
    args = vars(parser.parse_args())
    main(args)
