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
                             IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML    FP    FN  IDs   FM  MOTA  MOTP IDt  IDa IDm
UMA                         57.5% 67.9% 49.8% 56.2% 76.7% 517 158 250 109 18894 48325  387 1255 38.8% 0.230 242  160  91
JDE                         53.7% 71.6% 42.9% 48.8% 81.4% 517  93 254 170 12285 56539  512 2240 37.2% 0.220 152  297  45
MOTDT                       52.4% 67.5% 42.8% 52.1% 82.3% 517 118 255 144 12374 52882  457 1342 40.5% 0.208 203  287  69
SORT                        39.9% 55.1% 31.3% 47.5% 83.6% 517  76 247 194 10300 57921 1315 1231 37.0% 0.202 179 1075  35
DeepSORT(cos+g2)            52.0% 68.0% 42.1% 50.9% 82.3% 517 109 243 165 12094 54155  342 1216 39.7% 0.207 194  184  77
DeepSORTPlus(maha2+g3)      50.9% 66.5% 41.2% 50.9% 82.2% 517 109 242 166 12161 54174  366 1222 39.6% 0.207 228  193  86
DeepSORTPlus(maha2+g3+cam)  50.9% 66.4% 41.3% 51.1% 82.2% 517 112 243 162 12226 53979  367 1239 39.7% 0.207 231  192  89
DeepSORTPlus(maha3+g3)      51.0% 66.7% 41.3% 50.9% 82.2% 517 111 239 167 12164 54180  369 1226 39.6% 0.207 228  197  86
DeepSORTPlus(maha3+g3+cam)  51.4% 67.1% 41.7% 51.1% 82.2% 517 113 242 162 12216 53974  370 1246 39.7% 0.207 231  195  89
DeepSORTPlus(maha2+g2)      52.5% 68.7% 42.4% 50.7% 82.2% 517 106 244 167 12113 54396  332 1208 39.5% 0.206 207  177  83
DeepSORTPlus(maha2+g2+cam)  52.6% 68.8% 42.6% 50.9% 82.2% 517 109 243 165 12163 54228  332 1209 39.6% 0.206 207  177  83
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
                         IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP   FN  IDs  FM  MOTA  MOTP IDt  IDa IDm
UMA                     35.5% 33.9% 37.2% 96.3% 87.8% 64 64  0  0 3933 1104  898 513 79.9% 0.162 465  177   6
JDE                     88.4% 88.6% 88.0% 95.5% 96.5% 64 64  0  0 1038 1328  242 577 91.2% 0.165  31  104   1
MOTDT                   69.6% 71.4% 67.8% 91.2% 96.3% 64 62  2  0 1041 2609  230 807 86.9% 0.138  65  100   1
SORT                    23.7% 24.0% 23.4% 93.8% 96.3% 64 64  0  0 1078 1823 1417 694 85.4% 0.140  63 1168   0
DeepSORT(cos+g2)        74.9% 72.8% 76.9% 97.5% 92.6% 64 64  0  0 2318  735   94 432 89.3% 0.135  30   45   0
DeepSORTPlus(maha2+g2)  84.7% 83.6% 85.7% 97.2% 95.0% 64 64  0  0 1505  839   81 485 91.8% 0.133  39   26   0
DeepSORTPlus(cos+g3)    89.1% 88.1% 90.0% 97.0% 95.1% 64 64  0  0 1467  899   69 496 91.8% 0.132  23   26   0
DeepSORTPlus(maha3+g3)  87.3% 86.2% 88.2% 97.0% 95.1% 64 64  0  0 1485  872   57 487 91.8% 0.132  21   22   0
DeepSORTPlus(maha2+g3)  87.0% 86.0% 87.9% 97.1% 95.2% 64 64  0  0 1454  857   57 492 92.0% 0.132  23   21   1

DeepSORTPlus(Midas)     87.3% 86.2% 88.2% 97.0% 95.1% 64 64  0  0 1485  872 57 487 91.8% 0.132  21  22   0
DeepSORTPlus(maha2+g3)  87.0% 86.0% 87.9% 97.1% 95.2% 64 64  0  0 1454  857 57 492 92.0% 0.132  23  21   1
DeepSORTPlus(Lidar)     87.9% 86.7% 88.8% 97.2% 95.2% 64 64  0  0 1463  821 45 487 92.1% 0.132  19  19   1
```
> cos => cosine similarity to construct cost matrix  
> mahaX => softmax mahalanobis distance with degree of X cosine similarity to construct cost matrix  
> gX => gating matrix (chi-square testing) with X degree of freedom of kalman state


## NTU-MOTD Dataset Statistics
```
+-----------------+-----+--------------+--------+--------+---------+-------+---------+--------+--------+-----------+------------------------------------------------------------+
|     Sequence    | FPS |  Resolution  | Length | Tracks | Tr./Fr. | Boxes | Box/Fr. | Class  | Camera | Viewpoint |                         Conditions                         |
+-----------------+-----+--------------+--------+--------+---------+-------+---------+--------+--------+-----------+------------------------------------------------------------+
| 3p_da_pm_pp.msv |  30 | (1920, 1080) |  655   |  1594  |   2.4   |  1759 |   2.7   | person | static |   medium  |    indoor. 3 people and diverse appearance move around     |
|                 |     |              |        |        |         |       |         |        |        |           |  in a predicted motion pattern with less pose variation.   |
| 3p_da_pm_up.msv |  30 | (1920, 1080) |  675   |  1628  |   2.4   |  1780 |   2.6   | person | static |   medium  |    indoor. 3 people and diverse appearance move around     |
|                 |     |              |        |        |         |       |         |        |        |           |  in a predicted motion pattern with large pose variation.  |
| 3p_da_um_pp.msv |  30 | (1920, 1080) |  715   |  1694  |   2.4   |  1853 |   2.6   | person | static |   medium  |    indoor. 3 people and diverse appearance move around     |
|                 |     |              |        |        |         |       |         |        |        |           | in a unpredicted motion pattern with less pose variation.  |
| 3p_da_um_up.msv |  30 | (1920, 1080) |  643   |  1382  |   2.1   |  1572 |   2.4   | person | static |   medium  |    indoor. 3 people and diverse appearance move around     |
|                 |     |              |        |        |         |       |         |        |        |           | in a unpredicted motion pattern with large pose variation. |
| 3p_sa_pm_pp.msv |  30 | (1920, 1080) |  674   |  1598  |   2.4   |  1738 |   2.6   | person | static |   medium  |   indoor. 3 people with similiar appearance move around    |
|                 |     |              |        |        |         |       |         |        |        |           |  in a predicted motion pattern with less pose variation.   |
| 3p_sa_pm_up.msv |  30 | (1920, 1080) |  664   |  1566  |   2.4   |  1757 |   2.6   | person | static |   medium  |   indoor. 3 people with similiar appearance move around    |
|                 |     |              |        |        |         |       |         |        |        |           |  in a predicted motion pattern with large pose variation.  |
| 3p_sa_um_pp.msv |  30 | (1920, 1080) |  667   |  1569  |   2.4   |  1728 |   2.6   | person | static |   medium  |   indoor. 3 people with similiar appearance move around    |
|                 |     |              |        |        |         |       |         |        |        |           | in a unpredicted motion pattern with less pose variation.  |
| 3p_sa_um_up.msv |  30 | (1920, 1080) |  634   |  1502  |   2.4   |  1664 |   2.6   | person | static |   medium  |   indoor. 3 people with similiar appearance move around    |
|                 |     |              |        |        |         |       |         |        |        |           | in a unpredicted motion pattern with large pose variation. |
| 5p_da_pm_pp.msv |  30 | (1920, 1080) |  631   |  2145  |   3.4   |  2422 |   3.8   | person | static |   medium  |    indoor. 5 people and diverse appearance move around     |
|                 |     |              |        |        |         |       |         |        |        |           |  in a predicted motion pattern with less pose variation.   |
| 5p_da_pm_up.msv |  30 | (1920, 1080) |  636   |  2079  |   3.3   |  2444 |   3.8   | person | static |   medium  |    indoor. 5 people and diverse appearance move around     |
|                 |     |              |        |        |         |       |         |        |        |           |  in a predicted motion pattern with large pose variation.  |
| 5p_da_um_pp.msv |  30 | (1920, 1080) |  641   |  2128  |   3.3   |  2472 |   3.9   | person | static |   medium  |    indoor. 5 people and diverse appearance move around     |
|                 |     |              |        |        |         |       |         |        |        |           | in a unpredicted motion pattern with less pose variation.  |
| 5p_da_um_up.msv |  30 | (1920, 1080) |  648   |  2076  |   3.2   |  2516 |   3.9   | person | static |   medium  |    indoor. 5 people and diverse appearance move around     |
|                 |     |              |        |        |         |       |         |        |        |           | in a unpredicted motion pattern with large pose variation. |
| 5p_sa_pm_pp.msv |  30 | (1920, 1080) |  710   |  2192  |   3.1   |  2552 |   3.6   | person | static |   medium  |   indoor. 5 people with similiar appearance move around    |
|                 |     |              |        |        |         |       |         |        |        |           |  in a predicted motion pattern with less pose variation.   |
| 5p_sa_pm_up.msv |  30 | (1920, 1080) |  650   |  2148  |   3.3   |  2521 |   3.9   | person | static |   medium  |   indoor. 5 people with similiar appearance move around    |
|                 |     |              |        |        |         |       |         |        |        |           |  in a predicted motion pattern with large pose variation.  |
| 5p_sa_um_pp.msv |  30 | (1920, 1080) |  673   |  2186  |   3.2   |  2526 |   3.8   | person | static |   medium  |   indoor. 5 people with similiar appearance move around    |
|                 |     |              |        |        |         |       |         |        |        |           | in a unpredicted motion pattern with less pose variation.  |
| 5p_sa_um_up.msv |  30 | (1920, 1080) |  645   |  2050  |   3.2   |  2472 |   3.8   | person | static |   medium  |   indoor. 5 people with similiar appearance move around    |
|                 |     |              |        |        |         |       |         |        |        |           | in a unpredicted motion pattern with large pose variation. |
+-----------------+-----+--------------+--------+--------+---------+-------+---------+--------+--------+-----------+------------------------------------------------------------+
```

Preview of 5p\_sa\_um\_up.msv video sequence
[![preview](https://i.imgur.com/baL74Oy.png)](https://youtu.be/NfXUm-miaaU)


## MOT16 Train Detail Result
```
          IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT  ML    FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm
MOT16-02 39.7% 60.4% 29.6% 37.9% 77.4%  54   8  25  21  1978 11070  58   151 26.5% 0.212  25  35   3
MOT16-04 55.7% 77.4% 43.5% 49.0% 87.2%  83  12  45  26  3416 24276  58   379 41.6% 0.185  21  40   4
MOT16-05 57.7% 64.3% 52.3% 63.0% 77.4% 125  22  77  26  1251  2523  49   130 43.9% 0.254  66  17  41
MOT16-09 54.4% 55.1% 53.7% 73.0% 75.0%  25  13  10   2  1282  1419  25    61 48.1% 0.194  17  13   5
MOT16-10 51.2% 59.9% 44.7% 58.1% 78.0%  54  14  28  12  2025  5156  79   279 41.1% 0.251  40  34   8
MOT16-11 60.0% 66.0% 54.9% 66.5% 80.0%  69  24  20  25  1530  3073  19    45 49.6% 0.174   6  14   2
MOT16-13 48.1% 76.0% 35.2% 40.4% 87.3% 107  13  41  53   673  6823  35   162 34.2% 0.249  35  14  20
OVERALL  52.5% 68.8% 42.5% 50.8% 82.2% 517 106 246 165 12155 54340 323  1207 39.5% 0.207 210 167  83
```

## NTU-MOTD Test Detail Result
```
                 IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP  FN IDs   FM  MOTA  MOTP IDt IDa IDm
3p_da_pm_pp.msv 97.1% 96.2% 98.1% 98.1% 96.2%  3  3  0  0   62  31   0   19 94.2% 0.114   0   0   0
3p_da_pm_up.msv 90.0% 89.9% 90.0% 97.5% 97.3%  3  3  0  0   44  41   1   20 94.7% 0.126   0   1   0
3p_da_um_pp.msv 94.9% 94.2% 95.6% 98.5% 97.0%  3  3  0  0   51  25   2   13 95.4% 0.124   0   2   0
3p_da_um_up.msv 90.9% 89.4% 92.5% 98.8% 95.5%  3  3  0  0   65  17   2   12 93.9% 0.127   1   1   0
3p_sa_pm_pp.msv 97.7% 97.6% 97.7% 97.7% 97.6%  3  3  0  0   38  36   0   19 95.4% 0.114   0   0   0
3p_sa_pm_up.msv 84.7% 84.0% 85.4% 97.7% 96.0%  3  3  0  0   63  36   3   18 93.5% 0.132   1   2   0
3p_sa_um_pp.msv 93.2% 92.5% 93.9% 98.3% 96.9%  3  3  0  0   50  27   4   13 94.8% 0.118   2   1   0
3p_sa_um_up.msv 97.5% 96.5% 98.5% 98.5% 96.5%  3  3  0  0   53  23   0   10 94.9% 0.124   0   0   0
5p_da_pm_pp.msv 96.4% 96.1% 96.7% 96.7% 96.1%  5  5  0  0   84  71   0   37 92.8% 0.132   0   0   0
5p_da_pm_up.msv 96.0% 95.4% 96.5% 96.5% 95.4%  5  5  0  0   97  72   0   44 91.9% 0.137   0   0   0
5p_da_um_pp.msv 70.7% 70.1% 71.2% 97.4% 95.9%  5  5  0  0   89  55  10   24 92.8% 0.133   4   5   1
5p_da_um_up.msv 83.3% 81.7% 84.9% 97.1% 93.5%  5  5  0  0  140  60   6   31 90.1% 0.137   1   3   0
5p_sa_pm_pp.msv 93.8% 91.5% 95.3% 94.9% 92.1%  5  5  0  0  179 111   1   69 86.7% 0.136   0   1   0
5p_sa_pm_up.msv 84.6% 82.5% 85.1% 93.3% 92.3%  5  5  0  0  167 144   9   95 85.1% 0.144   1   4   0
5p_sa_um_pp.msv 64.4% 63.5% 65.4% 97.4% 94.6%  5  5  0  0  122  56  10   35 91.4% 0.147   8   0   0
5p_sa_um_up.msv 70.5% 68.9% 72.2% 97.5% 93.0%  5  5  0  0  150  52   9   33 89.7% 0.148   5   1   0
OVERALL         87.0% 86.0% 87.9% 97.1% 95.2% 64 64  0  0 1454 857  57  492 92.0% 0.132  23  21   1
```

## PLAN
Change object detection minimum threshold (0.8, 0.6, 0.4)
Change active tentative detection confidence (0.6, 0.8)
