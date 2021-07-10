import os
import os.path as osp
import argparse

import motmetrics as mm


def main(args):
    df_gt = mm.io.loadtxt(args['gt'])
    df_pred = mm.io.loadtxt(args['pred'])
    acc = mm.utils.compare_to_groundtruth(df_gt, df_pred, 'iou', distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics)
    print(mm.io.render_summary(summary,
                            namemap=mm.io.motchallenge_metric_names,
                            formatters=mh.formatters))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="mot2d groundtruth file")
    parser.add_argument("--pred", required=True, help="mot2d prediction file")
    args = vars(parser.parse_args())
    main(args)
