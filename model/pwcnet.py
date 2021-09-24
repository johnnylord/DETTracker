import time
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .correlation import correlation
except Exception as e:
    import os
    import os.path as osp
    import sys
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    from correlation import correlation


backwarp_grid = {}
backwarp_part = {}

def backwarp(feat, flow):
    if str(flow.shape) not in backwarp_grid:
        hori = torch.linspace(
                -1.0+(1.0/flow.shape[3]),
                1.0-(1.0/flow.shape[3]),
                flow.shape[3]
                ).view(1, 1, 1, -1).expand(-1, -1, flow.shape[2], -1)
        vert = torch.linspace(
                -1.0+(1.0/flow.shape[2]),
                1.0-(1.0/flow.shape[2]),
                flow.shape[2]
                ).view(1, 1, -1, 1).expand(-1, -1, -1, flow.shape[3])
        backwarp_grid[str(flow.shape)] = torch.cat([ hori, vert ], 1).cuda()

    if str(flow.shape) not in backwarp_part:
        backwarp_part[str(flow.shape)] = flow.new_ones([
                                            flow.shape[0],
                                            1,
                                            flow.shape[2],
                                            flow.shape[3]
                                            ])
    flow = torch.cat([
                flow[:, 0:1, :, :]/((feat.shape[3]-1.0)/2.0),
                flow[:, 1:2, :, :]/((feat.shape[2]-1.0)/2.0),
                ], 1)
    feat = torch.cat([ feat, backwarp_part[str(flow.shape)] ], 1)
    output = F.grid_sample(
                input=feat,
                grid=(backwarp_grid[str(flow.shape)]+flow).permute(0, 2, 3, 1),
                mode='bilinear', padding_mode='zeros', align_corners=False)
    mask = output[:, -1:, :, :]; mask[mask > 0.999] = 1.0; mask[mask < 1.0] = 0.0
    return output[:, :-1, :, :]*mask

class Extractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.netOne = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
        self.netTwo = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
        self.netThr = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
        self.netFou = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
        self.netFiv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
        self.netSix = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )

    def forward(self, x):
        c1 = self.netOne(x)     # (16, H/2, W/2)
        c2 = self.netTwo(c1)    # (32, H/4, W/4)
        c3 = self.netThr(c2)    # (64, H/8, W/8)
        c4 = self.netFou(c3)    # (96, H/16, W/16)
        c5 = self.netFiv(c4)    # (128, H/32, W/32)
        c6 = self.netSix(c5)    # (196, H/64, W/64)
        return [ c1, c2, c3, c4, c5, c6 ]


class Decoder(nn.Module):

    def __init__(self, intLevel):
        super().__init__()
        intPrevious = [
                None, None,
                81+32+2+2,
                81+64+2+2,
                81+96+2+2,
                81+128+2+2,
                81, None
                ][intLevel+1]
        intCurrent = [
                None, None,
                81+32+2+2,
                81+64+2+2,
                81+96+2+2,
                81+128+2+2,
                81, None
                ][intLevel+0]
        if intLevel < 6:
            self.netUpflow = nn.ConvTranspose2d(
                                in_channels=2,
                                out_channels=2,
                                kernel_size=4, stride=2, padding=1
                                )
            self.netUpfeat = nn.ConvTranspose2d(
                                in_channels=intPrevious+128+128+96+64+32,
                                out_channels=2,
                                kernel_size=4, stride=2, padding=1
                                )
            self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel+1]

        self.netOne = nn.Sequential(
                nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )
        self.netTwo = nn.Sequential(
                nn.Conv2d(in_channels=intCurrent+128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )
        self.netThr = nn.Sequential(
                nn.Conv2d(in_channels=intCurrent+128+128, out_channels=96, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )
        self.netFou = nn.Sequential(
                nn.Conv2d(in_channels=intCurrent+128+128+96, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )
        self.netFiv = nn.Sequential(
                nn.Conv2d(in_channels=intCurrent+128+128+96+64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )
        self.netSix = nn.Sequential(
                nn.Conv2d(in_channels=intCurrent+128+128+96+64+32, out_channels=2, kernel_size=3, stride=1, padding=1),
                )

    def forward(self, f1, f2, objPrevious):
        if objPrevious is None:
            flow = None
            feat = None
            start_time = time.time()
            volume = F.leaky_relu(
                        input=correlation.FunctionCorrelation(f1, f2),
                        negative_slope=0.1,
                        inplace=True
                        )
            feat = torch.cat([ volume ], 1)
        elif objPrevious is not None:
            flow = self.netUpflow(objPrevious['flow'])
            feat = self.netUpfeat(objPrevious['feat'])
            start_time = time.time()
            volume = F.leaky_relu(
                        input=correlation.FunctionCorrelation(f1, backwarp(f2, flow*self.fltBackwarp)),
                        negative_slope=0.1,
                        inplace=True
                        )
            feat = torch.cat([ volume, f1, flow, feat ], 1)

        feat = torch.cat([ self.netOne(feat), feat ], 1)
        feat = torch.cat([ self.netTwo(feat), feat ], 1)
        feat = torch.cat([ self.netThr(feat), feat ], 1)
        feat = torch.cat([ self.netFou(feat), feat ], 1)
        feat = torch.cat([ self.netFiv(feat), feat ], 1)
        flow = self.netSix(feat)

        return { 'flow': flow, 'feat': feat }


class Refiner(nn.Module):

    def __init__(self):
        super().__init__()
        self.netMain = nn.Sequential(
                nn.Conv2d(in_channels=81+32+2+2+128+128+96+64+32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1),
                )

    def forward(self, x):
        return self.netMain(x)


class PWCNet(nn.Module):
    PRETRAIN_URL = 'http://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch'

    def __init__(self):
        super().__init__()
        self.netExtractor = Extractor()
        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)
        self.netRefiner = Refiner()

        self.load_state_dict({
            strKey.replace('module', 'net'): weight
            for strKey, weight in torch.hub.load_state_dict_from_url(self.PRETRAIN_URL).items()
            })

    def forward(self, x1, x2):
        x1 = self.netExtractor(x1)
        x2 = self.netExtractor(x2)

        objPrevious = self.netSix(x1[-1], x2[-1], None)
        objPrevious = self.netFiv(x1[-2], x2[-2], objPrevious)
        objPrevious = self.netFou(x1[-3], x2[-3], objPrevious)
        objPrevious = self.netThr(x1[-4], x2[-4], objPrevious)
        objPrevious = self.netTwo(x1[-5], x2[-5], objPrevious)

        return objPrevious['flow'] + self.netRefiner(objPrevious['feat'])


def transform(img):
    arr = np.array(img)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    arr = np.ascontiguousarray(arr)
    norm = arr*(1.0/255.0)
    return torch.FloatTensor(norm)

def nptransform(frame):
    arr = frame.transpose(2, 0, 1).astype(np.float32)
    arr = np.ascontiguousarray(arr)
    norm = arr*(1.0/255.0)
    return torch.FloatTensor(norm)

if __name__ == "__main__":
    img1 = Image.open('./images/first.png')
    img2 = Image.open('./images/second.png')

    x1 = transform(img1)
    x2 = transform(img2)

    assert tuple(x1.shape) == tuple(x2.shape)

    imgWidth = x1.shape[2]
    imgHeight = x1.shape[1]
    alignWidth = int(math.floor(math.ceil(imgWidth/64.0)*64.0))
    alignHeight = int(math.floor(math.ceil(imgHeight/64.0)*64.0))

    device = torch.cuda.device(1)
    torch.cuda.set_device(device)

    model = PWCNet()
    model = model.cuda()
    model.eval()

    x1 = x1.cuda().view(1, 3, imgHeight, imgWidth)
    x2 = x2.cuda().view(1, 3, imgHeight, imgWidth)

    x1 = F.interpolate(input=x1, size=(alignHeight, alignWidth), mode='bilinear', align_corners=False)
    x2 = F.interpolate(input=x2, size=(alignHeight, alignWidth), mode='bilinear', align_corners=False)

    times = []
    for i in range(30):
        start_time = time.time()
        with torch.no_grad():
            flow = model(x1, x2)
        flow = 20 * F.interpolate(input=flow, size=(imgHeight, imgWidth), mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= float(imgWidth) / float(alignWidth)
        flow[:, 1, :, :] *= float(imgHeight) / float(alignHeight)
        # flow = flow.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        flow = flow.squeeze(0).permute(1, 2, 0)

        import cv2
        from utils import flow2img, torchflow2img
        # img = flow2img(flow)
        img = torchflow2img(flow)
        times.append(time.time()-start_time)
    # print(flow.min(), flow.max(), sum(times)/len(times), times)
    print(flow.min().item(), flow.max().item(), sum(times)/len(times), times)
    cv2.imwrite('flow.png', img)
