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

## Experiements

### Experiments with MOT16

#### MOT16 with DeepSORT and different detectors
```
          IDF1   IDP   IDR  Rcll  Prcn  GT MT   PT  ML    FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
DPM      32.3% 84.1% 20.0% 23.4% 98.5% 517 16  125 376   397 84573 116  736 22.9% 0.209  45  87  18
FRCNN    49.7% 81.6% 35.7% 41.0% 93.8% 517 45  229 243  3012 65130 130  956 38.2% 0.112  49 117  37
POI      49.9% 75.6% 37.2% 43.9% 89.2% 517 60  219 238  5845 61972 171 1124 38.4% 0.168 113 114  62
MRCNN    52.0% 68.0% 42.1% 50.9% 82.3% 517 109 243 165 12094 54155 342 1216 39.7% 0.207 194 184  77
```

#### MOT16 with different trackers but same detector (MRCNN)
```
                             IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML    FP    FN IDs   FM  MOTA  MOTP IDt IDa IDm
UMA                         57.5% 67.9% 49.8% 56.2% 76.7% 517 158 250 109 18894 48325 387 1255 38.8% 0.230 242 160  91
DeepSORT(cos+g2)            52.0% 68.0% 42.1% 50.9% 82.3% 517 109 243 165 12094 54155 342 1216 39.7% 0.207 194 184  77
DeepSORTPlus(maha2+g3)      50.9% 66.5% 41.2% 50.9% 82.2% 517 109 242 166 12161 54174 366 1222 39.6% 0.207 228 193  86
DeepSORTPlus(maha2+g3+cam)  50.9% 66.4% 41.3% 51.1% 82.2% 517 112 243 162 12226 53979 367 1239 39.7% 0.207 231 192  89
DeepSORTPlus(maha3+g3)      51.0% 66.7% 41.3% 50.9% 82.2% 517 111 239 167 12164 54180 369 1226 39.6% 0.207 228 197  86
DeepSORTPlus(maha3+g3+cam)  51.4% 67.1% 41.7% 51.1% 82.2% 517 113 242 162 12216 53974 370 1246 39.7% 0.207 231 195  89
DeepSORTPlus(maha2+g2)      52.5% 68.7% 42.4% 50.7% 82.2% 517 106 244 167 12113 54396 332 1208 39.5% 0.206 207 177  83
DeepSORTPlus(maha2+g2+cam)  52.6% 68.8% 42.6% 50.9% 82.2% 517 109 243 165 12163 54228 332 1209 39.6% 0.206 207 177  83
```
> The possible reason why deepsortplus is not superior to deepsort on MOT16 is that depth estimation in outdoor environments is less stable and accurate compared to indoor environments representing in NTU-MOTD.

### Experiments with NTU-MOTD

#### NTU-MOTD with DeepSORT and different detectors
```
          IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP   FN IDs   FM  MOTA  MOTP IDt IDa IDm
MRCNN    74.9% 72.8% 76.9% 97.5% 92.6% 64 64  0  0 2318  735  94  432 89.3% 0.135  30  45   0
YOLOv5   75.8% 76.6% 74.9% 95.1% 97.5% 64 64  0  0  728 1461  80  594 92.3% 0.101  40  34   2
```

#### NTU-MOTD with different trackers but same detector (MRCNN)
```
                         IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP  FN IDs   FM  MOTA  MOTP IDt IDa IDm
DeepSORT(cos+g2)        74.9% 72.8% 76.9% 97.5% 92.6% 64 64  0  0 2318 735  94  432 89.3% 0.135  30  45   0
DeepSORTPlus(maha2+g2)  84.7% 83.6% 85.7% 97.2% 95.0% 64 64  0  0 1505 839  81  485 91.8% 0.133  39  26   0
DeepSORTPlus(cos+g3)    89.1% 88.1% 90.0% 97.0% 95.1% 64 64  0  0 1467 899  69  496 91.8% 0.132  23  25   0
DeepSORTPlus(maha3+g3)  87.3% 86.2% 88.2% 97.0% 95.1% 64 64  0  0 1485 872  57  487 91.8% 0.132  21  22   0
DeepSORTPlus(maha2+g3)  87.0% 86.0% 87.9% 97.1% 95.2% 64 64  0  0 1454 857  57  492 92.0% 0.132  23  21   1
```
> cos => cosine similarity to construct cost matrix  
> mahaX => softmax mahalanobis distance with degree of X cosine similarity to construct cost matrix  
> gX => gating matrix (chi-square testing) with X degree of freedom of kalman state
