import os
import os.path as osp
import argparse


def main(args):
    tree = os.walk(args['input'])

    inputs = []
    for path, dirs, files in tree:
        targets = [ osp.join(path, f) for f in files if '.txt' in f ]
        if len(targets) > 0:
            inputs.extend(targets)

    outputs = [ osp.join(args['output'], osp.basename(i).replace('.', f"-{args['detector']}.").replace('16', '17'))
                for i in inputs ]

    if not osp.exists(args['output']):
        os.makedirs(args['output'])

    for i, o in zip(inputs, outputs):
        with open(i, 'r') as f:
            content = f.read()
        with open(o, 'w') as f:
            f.write(content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector")
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = vars(parser.parse_args())
    main(args)
