import os
import os.path as osp
import configparser

import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T


class MOTSequence:
    """MOT Sequences targeted at Pedestrians (Ignore other classes)

    Dataset Properties:
        - Moving or static cmaera
        - Different viewpoints
        - Different conditions (weather)
    """
    GT_COLUMNS = [ "Frame", "Track", "Xmin", "Ymin", "Width", "Height", "Ignore", "Type", "Visibility" ]
    DET_COLUMNS = [ "Frame", "Track", "Xmin", "Ymin", "Width", "Height", "Conf", "X", "Y", "Z" ]

    DETECTOR_TABLE = {
        'default': 'det.txt',
        'sdp': 'det-sdp.txt',
        'dpm': 'det-dpm.txt',
        'faster-rcnn': 'det-faster-rcnn.txt',
        'processed': 'det-processed.txt',
        'sdp-processed': 'det-sdp-processed.txt',
        'dpm-processed': 'det-dpm-processed.txt',
        'faster-rcnn-processed': 'det-faster-rcnn-processed.txt',
    }
    def __init__(
            self,
            root,
            mode='train',
            useextra=False,
            detector='default',
            min_visibility=0.5,
            min_conf_threshold=0.5):
        self.root = root
        self.mode = mode
        self.useextra = useextra
        self.detector = detector
        self.min_visibility = min_visibility
        self.min_conf_threshold = min_conf_threshold
        self.imgTransform = T.ToTensor()

        # Sanity Check
        assert (mode == 'train' or mode =='test')
        assert (
            detector == 'default'                   # MOT16
            or detector == 'sdp'                    # MOT17-SDP
            or detector == 'dpm'                    # MOT17-DPM
            or detector == 'faster-rcnn'            # MOT17-FRCNN
            or detector == 'processed'              # MOT16
            or detector == 'sdp-processed'          # MOT17-SDP (ReID)
            or detector == 'dpm-processed'          # MOT17-DPM (ReID)
            or detector == 'faster-rcnn-processed'  # MOT17-FRCNN (ReID)
            )

        # Read seqinfo.ini
        seqpath = osp.join(root, 'seqinfo.ini')
        parser = configparser.ConfigParser()
        parser.read(seqpath)

        # All images
        imgDir = osp.join(root, 'img1')
        self.imgs = sorted([ osp.join(imgDir, f) for f in os.listdir(imgDir) ])

        # Read ground truth tracks
        if mode == 'train':
            gtfile = osp.join(root, 'gt', 'gt.txt')
            self.gt_df = pd.read_csv(gtfile, header=None, names=MOTSequence.GT_COLUMNS)

        # Read detection
        detfile = osp.join(root, 'det', MOTSequence.DETECTOR_TABLE[detector])
        self.det_df = pd.read_csv(detfile, header=None, names=MOTSequence.DET_COLUMNS)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Read Image
        imgPath = self.imgs[idx]
        img = Image.open(imgPath)
        img = self.imgTransform(img)

        # Filter out groundtruth trajectory with respect to the image
        frameId = int(osp.basename(imgPath).split(".")[0])
        if self.mode == 'train':
            tracks = self.gt_df.loc[(
                            (self.gt_df['Frame'] == frameId)
                            & (self.gt_df['Ignore'] == 1)
                            & (
                                (self.gt_df['Type'] == 1)
                                | (self.gt_df['Type'] == 2)
                            )
                            & (self.gt_df['Visibility'] >= self.min_visibility)
                        )]
            names = [ "Track", "Xmin", "Ymin", "Width", "Height", "Visibility" ]
            tboxes = tracks[names]

        # Filter out detections with respect to the image
        dets = self.det_df.loc[(
                        (self.det_df['Frame'] == frameId)
                        & (self.det_df['Conf'] >= self.min_conf_threshold)
                    )]
        names = [ "Track", "Xmin", "Ymin", "Width", "Height", "Conf" ]
        bboxes = dets[names]

        if self.mode == 'train':
            return img, tboxes.to_numpy().tolist(), bboxes.to_numpy().tolist()
        else:
            return img, [], bboxes.to_numpy().tolist()


class MOTDSequence(MOTSequence):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # All depth maps
        midasDir = osp.join(self.root, 'midas')
        self.depthmaps = sorted([ osp.join(midasDir, f) for f in os.listdir(midasDir) ])
        # All flow maps
        flowDir = osp.join(self.root, 'flow')
        self.flows = sorted([ osp.join(flowDir, f) for f in os.listdir(flowDir) ])
        # Sanity Check
        for img, depthmap, flow in zip(self.imgs, self.depthmaps, self.flows):
            imgFile = int(osp.basename(img).split(".")[0])
            mapFile = int(osp.basename(depthmap).split(".")[0])
            floFile = int(osp.basename(flow).split(".")[0])
            assert imgFile == mapFile == floFile

    def __getitem__(self, idx):
        img, tboxes, bboxes = super().__getitem__(idx)

        # Extract depth map
        mapPath = self.depthmaps[idx]
        depthmap = Image.open(mapPath)
        depthmap = self.imgTransform(depthmap)

        # Extract flow map
        floPath = self.flows[idx]
        flow = self._readflow(floPath)
        flow = torch.tensor(flow)

        return img, depthmap, flow, tboxes, bboxes

    def _readflow(self, floPath):
        """Read .flo file in middlebury format"""
        with open(floPath, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                raise RuntimeError("magic header in .flo is corrupted")
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
        return np.resize(data, (int(h), int(w), 2))


if __name__ == "__main__":
    print("Normal Sequence")
    sequence = MOTSequence(root="/home/johnnylord/dataset/MOT16/train/MOT16-02")
    img, tboxes, bboxes = sequence[3]
    print(img.shape)
    print(tboxes)
    print(bboxes)
    print("="*30)

    print("Depth Sequence")
    sequence = MOTDSequence(root="/home/johnnylord/dataset/MOT16/train/MOT16-02")
    img, depthmap, flow, tboxes, bboxes = sequence[3]
    print(img.shape)
    print(depthmap.shape)
    print(flow.shape)
    print(tboxes)
    print(bboxes)
    print("="*30)

    import time
    start_time = time.time()
    for i in range(len(sequence)):
        print(f"Read:{i}", end='\r\b')
        sample = sequence[i]
    print("Elapsed:", time.time()-start_time)
