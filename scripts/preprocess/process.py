import os
import os.path as osp
import sys
import subprocess


MOT_DIR = "/home/johnnylord/dataset/MOT16/test/"

SEQUENCES = [ osp.join(MOT_DIR, seq) for seq in os.listdir(MOT_DIR) ]


for seq in SEQUENCES:
    cmdline = " ".join([
        sys.executable, "-m", "scripts.preprocess.addmask",
        "--sequence", seq,
        "--reid1", "run/motreid/best.pth",
        "--reid2", "run/market1501/best.pth"
        ])
    proc = subprocess.Popen(cmdline, shell=True)
    proc.wait()
    proc.terminate()
