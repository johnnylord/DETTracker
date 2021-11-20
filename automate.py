import os
import os.path as osp
import sys
import subprocess


configs = [
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.3),
    (0.5, 0.3, 0.5),
    (0.3, 0.5, 0.5),
    (0.3, 0.5, 0.3),
    (0.5, 0.3, 0.3),
    (0.3, 0.3, 0.5),
    (0.3, 0.3, 0.3),
    ]

for mahacos, cos, iou in configs:
    cmdline = " ".join([
        sys.executable, "runall.py",
        "--mot_dir", "/home/johnnylord/dataset/MOT/NTU-MOTD/test",
        "--tracker", "DeepSORTPlus",
        "--detector", "mrcnn-processed-mask",
        "--min_obj_conf", "0.8",
        "--degree", "4",
        "--n_lost", "3",
        "--iou_dist_threshold", f"{iou}",
        "--cos_dist_threshold", f"{cos}",
        "--maha_cos_dist_threshold", f"{mahacos}",
        "--guess_limit", "1",
        "--indoor",
        "--output", f"experiments/output/DET/NTU-MOTD_longdata_{mahacos}_{cos}_{iou}"
        ])
    proc = subprocess.Popen(cmdline, shell=True)
    proc.wait()
    proc.terminate()
