import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def flow2img(flow):
    """Convert optical flow into color image"""
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    UNKOWN_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKOWN_FLOW_THRESHOLD
    pr2 = abs(v) > UNKOWN_FLOW_THRESHOLD
    idx_unknown = (pr1|pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # Get max value in each direction
    maxu, maxv = -999., -999.
    minu, minv = 999., 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)
    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0
    return np.uint8(img)

def torchflow2img(flow):
    """Convert optical flow into color image"""
    u = flow[..., 0]
    v = flow[..., 1]

    UNKOWN_FLOW_THRESHOLD = 1e7
    pr1 = torch.abs(u) > UNKOWN_FLOW_THRESHOLD
    pr2 = torch.abs(v) > UNKOWN_FLOW_THRESHOLD
    idx_unknown = (pr1|pr2)
    u[idx_unknown] = 0.
    v[idx_unknown] = 0.

    # Get max value in each direction
    maxu, maxv = -999., -999.
    minu, minv = 999., 999.
    maxu = max(maxu, torch.max(u).item())
    maxv = max(maxv, torch.max(v).item())
    minu = min(minu, torch.min(u).item())
    minv = min(minv, torch.min(v).item())

    rad = torch.sqrt(u**2 + v**2)
    maxrad = max(-1, torch.max(rad).item())
    u = u / (maxrad+torch.finfo(torch.float32).eps)
    v = v / (maxrad+torch.finfo(torch.float32).eps)

    img = torch_compute_color(u, v)
    idx = idx_unknown.unsqueeze(-1).repeat(1, 1, 3)
    img[idx] = 0
    return np.uint8(img.detach().cpu().numpy())

def compute_color(u, v):
    """Compute optical flow color map"""
    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols-1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk-k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f)*col0 + f*col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255*col*(1-NAN_idx)))

    return img

def torch_compute_color(u, v):
    """Compute optical flow color map"""
    height, width = u.size(0), u.size(1)
    img = torch.zeros((height, width, 3)).to(u.device)

    NAN_idx = torch.isnan(u) | torch.isnan(v)
    u[NAN_idx] = 0.
    v[NAN_idx] = 0.

    colorwheel = torch.FloatTensor(make_color_wheel()).to(u.device)
    ncols = colorwheel.size(0)

    rad = torch.sqrt(u**2 + v**2)
    a = torch.atan2(-v, -u) / 3.141592

    fk = (a+1) / 2 * (ncols-1) + 1
    k0 = torch.floor(fk).long()
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk-k0

    for i in range(0, colorwheel.size(1)):
        tmp = colorwheel[..., i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f)*col0 + f*col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = torch.logical_not(idx)

        col[notidx] *= 0.75
        img[..., i] = torch.floor(255*col*(1-NAN_idx.long()))

    return img

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel

