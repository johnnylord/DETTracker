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

## Experiment Result on NTU-MOTD

### Different ReID Model (ReID=[market1501|motreid])
```
                     IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML  FP   FN IDs   FM  MOTA  MOTP IDt IDa IDm
DeepSORT(Mar+Res50) 74.5% 75.4% 73.4% 94.9% 97.7% 64 64  0  0 655 1503  94  598 92.4% 0.101  49  33   0
DeepSORT(MOT+Res50) 75.8% 76.6% 74.9% 95.1% 97.5% 64 64  0  0 728 1461  80  594 92.3% 0.101  40  34   2
```

### Different Tracker (Tracker=[deepsort|deepsortplus])
TODO

## Experiment Result on MOT16

### Different ReID Model (ReID=[market1501|motreid])
TODO

### Different Tracker (Tracker=[deepsort|deepsortplus])
TODO
