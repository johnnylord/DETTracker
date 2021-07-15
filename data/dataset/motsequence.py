import os
import os.path as osp
import configparser

import cv2
import numpy as np
import pandas as pd
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
    MOT16DET_COLUMNS = [ "Frame", "Track", "Xmin", "Ymin", "Width", "Height", "Conf", "X", "Y", "Z" ]
    MOT17DET_COLUMNS = [ "Frame", "Track", "Xmin", "Ymin", "Width", "Height", "Conf" ]

    DETECTOR_TABLE = {
        # =========== Detection Only ===========
        'default': 'det.txt',
        'default-processed': 'det-processed.txt',
        'default-processed-mask': 'det-processed-mask.txt',
        'default-processed-mask-all': 'det-processed-mask-all.txt',
        'default-processed-market1501': 'det-processed-market1501.txt',
        'default-processed-market1501-mask': 'det-processed-market1501-mask.txt',
        'default-processed-market1501-mask-all': 'det-processed-market1501-mask-all.txt',
    }
    GT_TABLE = {
        # =========== Detection Only ===========
        'default': 'gt.txt',
        'default-processed': 'gt.txt',
        'default-processed-mask': 'gt.txt',
        'default-processed-mask-all': 'gt-frcnn.txt',
        'default-processed-market1501': 'gt.txt',
        'default-processed-market1501-mask': 'gt.txt',
        'default-processed-market1501-mask-all': 'gt-frcnn.txt',
    }
    def __init__(
            self,
            root,
            mode='train',
            detector='default',
            min_visibility=0.5,
            min_conf_threshold=0.0):
        self.root = root
        self.mode = mode
        self.detector = detector
        self.min_visibility = min_visibility
        self.min_conf_threshold = min_conf_threshold
        self.imgTransform = T.ToTensor()

        # Sanity Check
        assert (mode == 'train' or mode =='test')
        assert (
            detector == 'default'
            or detector == 'default-processed'
            or detector == 'default-processed-mask'
            or detector == 'default-processed-mask-all'
            or detector == 'default-processed-market1501'
            or detector == 'default-processed-market1501-mask'
            or detector == 'default-processed-market1501-mask-all'
            )

        # Read seqinfo.ini
        seqpath = osp.join(root, 'seqinfo.ini')
        parser = configparser.ConfigParser()
        parser.read(seqpath)
        self.name = parser['Sequence']['name']
        self.fps = int(parser['Sequence']['framerate'])
        self.imgWidth = int(parser['Sequence']['imwidth'])
        self.imgHeight = int(parser['Sequence']['imheight'])

        # All images
        imgDir = osp.join(root, 'img1')
        self.imgs = sorted([ osp.join(imgDir, f) for f in os.listdir(imgDir) ])

        # Read ground truth tracks
        if mode == 'train':
            gtfile = osp.join(root, 'gt', MOTSequence.GT_TABLE[detector])
            df = pd.read_csv(gtfile, header=None, names=MOTSequence.GT_COLUMNS)
            # Process Groundtruth
            self.all_tboxes = {}
            for imgFile in self.imgs:
                frameId = int(osp.basename(imgFile).split(".")[0])
                tracks = df.loc[(
                                (df['Frame'] == frameId)
                                & (df['Ignore'] == 1)
                                & (
                                    (df['Type'] == 1)
                                    | (df['Type'] == 2)
                                )
                                & (df['Visibility'] >= self.min_visibility)
                            )]
                names = [ "Track", "Xmin", "Ymin", "Width", "Height", "Visibility" ]
                tboxes = tracks[names].to_numpy()
                if len(tboxes) > 0:
                    tboxes[:, 1] -= 1
                    tboxes[:, 2] -= 1
                tboxes = tboxes.tolist()
                self.all_tboxes[frameId] = tboxes

        # Read detection
        detfile = osp.join(root, 'det', MOTSequence.DETECTOR_TABLE[detector])
        df = pd.read_csv(detfile, header=None)
        if (
            detector == 'sdp'
            or detector == 'dpm'
            or detector == 'frcnn'
        ):
            columns = MOTSequence.MOT17DET_COLUMNS + list(df.columns[len(MOTSequence.MOT17DET_COLUMNS):])
            df.columns = columns
            extra_cols = columns[len(MOTSequence.MOT17DET_COLUMNS):]
        else:
            columns = MOTSequence.MOT16DET_COLUMNS + list(df.columns[len(MOTSequence.MOT16DET_COLUMNS):])
            df.columns = columns
            extra_cols = columns[len(MOTSequence.MOT16DET_COLUMNS):]

        # Process Detection
        self.all_masks = {}
        self.all_bboxes = {}
        for imgFile in self.imgs:
            frameId = int(osp.basename(imgFile).split(".")[0])
            dets = df.loc[(
                            (df['Frame'] == frameId)
                            & (df['Conf'] >= self.min_conf_threshold)
                        )]
            names = [ "Track", "Xmin", "Ymin", "Width", "Height", "Conf" ]
            if 'mask' not in detector:
                names += extra_cols
            else:
                names += extra_cols[:-1]
            bboxes = dets[names].to_numpy()
            if (
                len(bboxes) > 0
                and (
                    detector == 'default'
                    or detector == 'sdp'
                    or detector == 'dpm'
                    or detector == 'frcnn'
                )
            ):
                bboxes[:, 1] -= 1
                bboxes[:, 2] -= 1
            bboxes = bboxes.tolist()
            self.all_bboxes[frameId] = bboxes

            if 'mask' not in detector:
                self.all_masks[frameId] = []
                continue

            self.all_masks[frameId] = []
            for mask_name in dets[extra_cols[-1]]:
                mask_path = osp.join(osp.dirname(detfile), mask_name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                self.all_masks[frameId].append(mask)

    def __str__(self):
        content = (
            f"[Sequence]\n"
            f" - name: {self.name}\n"
            f" - fps: {self.fps}\n"
            f" - width: {self.imgWidth}\n"
            f" - height: {self.imgHeight}\n"
            f" - length: {self.__len__()}\n"
            f" - detector: {self.detector}\n"
            f" - min_visibility: {self.min_visibility}\n"
            f" - min_conf_threshold: {self.min_conf_threshold}\n"
        )
        return content

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Read Image
        imgPath = self.imgs[idx]
        img = Image.open(imgPath)
        img = self.imgTransform(img)

        # Filter out groundtruth trajectory with respect to the image
        frameId = int(osp.basename(imgPath).split(".")[0])
        tboxes = self.all_tboxes[frameId] if self.mode == 'train' else []
        bboxes = self.all_bboxes[frameId]
        masks = self.all_masks[frameId]
        return img, tboxes, bboxes, masks


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
        img, tboxes, bboxes, masks = super().__getitem__(idx)

        # Extract depth map
        mapPath = self.depthmaps[idx]
        depthmap = Image.open(mapPath)
        depthmap = self.imgTransform(depthmap)

        # Extract flow map
        floPath = self.flows[idx]
        flow = self._readflow(floPath)
        flow = torch.tensor(flow)

        return img, depthmap, flow, tboxes, bboxes, masks

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
    # print("Normal Sequence")
    # sequence = MOTSequence(root="/home/johnnylord/dataset/MOT16/train/MOT16-02", detector='frcnn', mode='train')
    # img, tboxes, bboxes = sequence[3]
    # print(img.shape)
    # print(tboxes)
    # print(bboxes)
    # print("="*30)

    print("Depth Sequence")
    sequence = MOTDSequence(root="/home/johnnylord/dataset/MOT16/train/MOT16-02", detector='frcnn-processed', mode='train')
    img, depthmap, flowmap, tboxes, bboxes, masks = sequence[3]
    print(img.shape, img.min(), img.max())
    print(depthmap.shape, depthmap.min(), depthmap.max())
    print(flowmap.shape, flowmap.min(), flowmap.max())
    print(bboxes[0])

    # print(tboxes)
    # print(bboxes)
    # print("="*30)

    # import time
    # start_time = time.time()
    # for i in range(len(sequence)):
        # print(f"Read:{i}", end='\r\b')
        # sample = sequence[i]
    # print("Elapsed:", time.time()-start_time)
