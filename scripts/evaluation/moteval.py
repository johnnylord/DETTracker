import os
import os.path as osp
import argparse

import motmetrics as mm


def main(args):
    gts = sorted([ osp.join(args['mot_gt'], seq) for seq in os.listdir(args['mot_gt']) ])
    pds = sorted([ osp.join(args['mot_pd'], seq) for seq in os.listdir(args['mot_pd']) ])

    accs = []
    for seq_gt, seq_pd in zip(gts, pds):
        file_gt = osp.join(seq_gt, 'gt', 'gt.txt' if len(args['suffix']) == 0 else f"gt-{args['suffix']}.txt" )
        file_pd = osp.join(seq_pd, f'{osp.basename(seq_pd)}.txt')

        df_gt = mm.io.loadtxt(file_gt)
        if 'MOT16' in file_gt:
            df_gt = df_gt.loc[(
                            (df_gt['ClassId'] == 1)     # Pedetrain class
                            | (df_gt['ClassId'] == 2)   # People on vehicles
                            & (df_gt['Confidence'] == 1)# Considered when evaluation
                        )]
        df_pred = mm.io.loadtxt(file_pd)
        acc = mm.utils.compare_to_groundtruth(df_gt, df_pred, 'iou', distth=0.5)
        accs.append(acc)

    names = [ osp.basename(pd) for pd in pds ]
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names,
                            generate_overall=True,
                            metrics=mm.metrics.motchallenge_metrics)
    print(mm.io.render_summary(summary,
                            namemap=mm.io.motchallenge_metric_names,
                            formatters=mh.formatters))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot_gt", required=True, help="mot2d groundtruth sequence directory")
    parser.add_argument("--mot_pd", required=True, help="mot2d prediction sequence directory")
    parser.add_argument("--suffix", default="", help="suffix of gt file")
    args = vars(parser.parse_args())
    main(args)
