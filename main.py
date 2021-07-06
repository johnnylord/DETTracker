import os
import os.path as osp
import sys
import subprocess


INPUT_DIR = osp.expanduser("~/dataset/dataset/")
TARGET_DIR = "NTU-MOTD/test/"

sequence_dirs = [ osp.join(INPUT_DIR, seq) for seq in os.listdir(INPUT_DIR) ]
procs = []
for seq_dir in sequence_dirs:
    seq_name = osp.basename(seq_dir)
    print(f"Process {seq_name}")
    cmdline = " ".join([
        sys.executable, "ntu2mot.py",
        "--input", seq_dir,
        "--output", osp.join(TARGET_DIR, seq_name)
    ])
    proc = subprocess.Popen(cmdline, shell=True, stdout=subprocess.PIPE)
    procs.append(proc)

for proc in procs:
    proc.wait()
    proc.terminate()
