import os
import os.path as osp
import sys
import subprocess


INPUT_DIR = osp.expanduser("download/MOT16/test/")

sequence_dirs = [ osp.join(INPUT_DIR, seq) for seq in os.listdir(INPUT_DIR) ]

procs = []
for seq_dir in sequence_dirs:
    seq_name = osp.basename(seq_dir)
    print(f"Process {seq_name}")
    cmdline = " ".join([
        sys.executable, "addmidas.py",
        "--sequence", seq_dir,
    ])
    proc = subprocess.Popen(cmdline, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    proc.terminate()
