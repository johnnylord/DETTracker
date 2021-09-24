import os
import sys
f = open(os.devnull, 'w')
import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from ptflops import get_model_complexity_info
from model.pwcnet import PWCNet, transform
from model.resnet import resnet50_reid
from model.scene import SceneDetector


# MIDAS
# =====================================================================
model_type = "DPT_Large"
with torch.cuda.device(0):
    net = torch.hub.load('intel-isl/MiDaS', model_type)
    macs, params = get_model_complexity_info(net, (3, 384, 672), ost=f)
    print("MIDAS")
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

print()

# Mask-RCNN
# =====================================================================
with torch.cuda.device(0):
    net = maskrcnn_resnet50_fpn(pretrained=True)
    macs, params = get_model_complexity_info(net, (3, 1080, 1920), ost=f)
    print("MASK_RCNN")
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

print()

# PWCNet
# =====================================================================
def input_constructor(resolution):
    x1 = torch.rand(*resolution).cuda()
    x2 = torch.rand(*resolution).cuda()
    return { 'x1': x1, 'x2': x2 }

with torch.cuda.device(0):
    net = PWCNet().cuda()
    macs, params = get_model_complexity_info(net, (1, 3, 1024, 1920),
                                            input_constructor=input_constructor,
                                            ost=f)
    print("PWCNET MODEL")
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


print()

# ReID
# =====================================================================
with torch.cuda.device(0):
    net = resnet50_reid(features=128, classes=1)
    macs, params = get_model_complexity_info(net, (3, 256, 128), ost=f)
    print("REID MODEL")
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


print()

# Scene Detector
# =====================================================================
with torch.cuda.device(0):
    net = SceneDetector()
    macs, params = get_model_complexity_info(net, (1, 224, 224), ost=f)
    print("Scene MODEL")
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
