# DETTracker

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
