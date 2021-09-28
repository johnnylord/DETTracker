import os
import os.path as osp
import argparse
import random


def main(args):
    midas_dirs = [ osp.join(args['dataset'], seq, 'img1') for seq in os.listdir(args['dataset']) ]

    # Collect all depth maps
    all_files = []
    for midas in midas_dirs:
        files = sorted([ osp.join(midas, f) for f in os.listdir(midas) ])
        all_files += files

    # Sample depth maps
    random.shuffle(all_files)
    samples = all_files[:args['sample']]

    # Copy depth maps to output directory
    for idx, sample in enumerate(samples):
        with open(sample, 'rb') as f:
            content = f.read()

        output = osp.join(args['output'], f"{idx}.jpg")
        if not osp.exists(osp.dirname(output)):
            os.makedirs(osp.dirname(output))

        with open(output, 'wb') as f:
            f.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="MOT Dataset")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--sample", default=4000, help="Output directory")
    args = vars(parser.parse_args())
    main(args)
