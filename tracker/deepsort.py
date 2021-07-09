import torch
import torch.nn.functional as F


class DeepSORTPlus:

    def __init__(self):
        pass

    def __call__(self, img, depthmap, flowmap, bboxes):
        """Perform tracking on per frame basis

        Argument:
            img (tensor): tensor of shape (3, H, W)
            depthmap (tensor): tensor of shape (3, H, W)
            flowmap (tensor): tensor of shape (H, W, 2)
            bboxes (list): list of bounding boxes

        Format of depthmap:
            The value range of depth map is between 0 and 1, you can multiply
            a factor to get a relational estimated depth value.

        Format of flowmap:
            The value range of flow map is unbounded. In each pixel, there is
            a 2D xy pixel offset vector between consecutive frame.

        Format of bboxes:
            Each box in bboxes is represented as:
                (trackId, xmin, ymin, width, height, conf, 128 dim features...)
            (xmin, ymin , width, height) is in pixel coordinate

        Return:
            All objects being tracked represented as tracks. Each track has
            following information:
                1. Track ID
                2. bounding box
        """
        pass
