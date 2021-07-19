# deepsortplus

## Download Dataset
```bash
$ wget -O download/MOT16.tar.gz
$ wget -O download/NTU-MOTD.tar.gz
```

## Train ReID Model
```bash
# Train Resnet50 ReID Model on different dataset
$ python3 main.py --config config/motreid.yml
$ python3 main.py --config config/market1501.yml
```

## Run Tracker on MOT Dataset
```bash
$ python3 runall.py \
    --mot_dir [MOT Dataset Directory] \
    --detector [detector to use] \
    --tracker [tracker to use] \
    # tracker options can used (See run.py)
```

## Evaluate Result on MOT Dataset
```bash
$ python3 scripts/evaluation/moteval.py \
    --mot_gt [MOT Dataset Directory] \
    --mot_pd [MOT Prediction directory]
```

## Visualize Video Result with MPV (2x2 Layout)
1. Play sequence data and export it as videos
```bash
$ python3 scripts/play.py \
    --sequence [sequence directory] \
    --silent \
    --export
```
2. Visualize exported videos
```bash
$ mpv gt.mp4 \
    --external-file=det.mp4 \
    --external-file=map.mp4 \
    --external-file=flo.mp4 \
    --lavfi-complex="[vid1][vid2]hstack=inputs=2[top];[vid3][vid4]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[vo]"
```

## MOT16 with DeepSORT and different detectors
```
          IDF1   IDP   IDR  Rcll  Prcn  GT MT   PT  ML    FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
DPM      32.3% 84.1% 20.0% 23.4% 98.5% 517 16  125 376   397 84573 116  736 22.9% 0.209  45  87  18
FRCNN    49.7% 81.6% 35.7% 41.0% 93.8% 517 45  229 243  3012 65130 130  956 38.2% 0.112  49 117  37
POI      49.9% 75.6% 37.2% 43.9% 89.2% 517 60  219 238  5845 61972 171 1124 38.4% 0.168 113 114  62
MRCNN    52.0% 68.0% 42.1% 50.9% 82.3% 517 109 243 165 12094 54155 342 1216 39.7% 0.207 194 184  77
```

## NTU-MOTD with DeepSORT and different detectors
```
          IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP   FN IDs   FM  MOTA  MOTP IDt IDa IDm
MRCNN    74.9% 72.8% 76.9% 97.5% 92.6% 64 64  0  0 2318  735  94  432 89.3% 0.135  30  45   0
YOLOv5   75.8% 76.6% 74.9% 95.1% 97.5% 64 64  0  0  728 1461  80  594 92.3% 0.101  40  34   2
```

## MOT16 with different trackers but same detectors
```

```

## NTU-MOTD with different trackers but same detector (MRCNN)
```
                 IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP  FN IDs   FM  MOTA  MOTP IDt IDa IDm
DeepSORT        74.9% 72.8% 76.9% 97.5% 92.6% 64 64  0  0 2318 735  94  432 89.3% 0.135  30  45   0
DeepSORTPlus    85.9% 84.9% 86.7% 96.9% 95.1% 64 64  0  0 1470 905  64  501 91.7% 0.132  26  27   0
```
