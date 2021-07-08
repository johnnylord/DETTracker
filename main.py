import os
import os.path as osp
import sys
import subprocess
import multiprocessing

NUM_WORKERS = 3
MOT_DIR = "/home/johnnylord/dataset/MOT16/test"

def worker(subsequences):
    for sequence in subsequences:
        cmdline = " ".join([
            sys.executable, "scripts/play.py",
            "--sequence", sequence,
            "--silent", "--export"
        ])
        child = subprocess.Popen(cmdline, shell=True, stdout=subprocess.PIPE)
        child.wait()
        child.terminate()

sequences = [ osp.join(MOT_DIR, seq) for seq in os.listdir(MOT_DIR) ]
n_seqs_per_worker = len(sequences)//NUM_WORKERS

subtasks = []
for i in range(NUM_WORKERS):
    start = i*n_seqs_per_worker
    end = i*n_seqs_per_worker+n_seqs_per_worker
    task = sequences[start:end]
    subtasks.append(task)

if len(sequences)%NUM_WORKERS != 0:
    subtasks[-1].extend(sequences[end:])

procs = []
for subsequences in subtasks:
    proc = multiprocessing.Process(target=worker, args=(subsequences,), daemon=True)
    proc.start()
    procs.append(proc)

for proc in procs:
    proc.join()
