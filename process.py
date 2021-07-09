import os
import os.path as osp
import sys
import subprocess


INPUT_DIR = osp.expanduser("~/dataset/MOT16/train/")
sequence_dirs = [ osp.join(INPUT_DIR, seq) for seq in os.listdir(INPUT_DIR) ]
for seq_dir in sequence_dirs:
    seq_name = osp.basename(seq_dir)
    print(f"Process {seq_name}")
    cmdline = " ".join([
        sys.executable, "scripts/preprocess/addreid.py",
        "--sequence", seq_dir,
        "--reid", "run/motreid/best.pth",
    ])
    proc = subprocess.Popen(cmdline, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    proc.terminate()
