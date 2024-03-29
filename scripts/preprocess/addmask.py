import os
import os.path as osp
import argparse

import GPUtil
from tqdm import tqdm
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn

from data.dataset.motreid import MOTreID
from data.dataset.motsequence import MOTSequence
from model.resnet import resnet50_reid

LOOKUP = {
    # Reid with MOT
    # 'default-processed': 'det-processed-mask.txt',
    # 'frcnn-processed': 'det-frcnn-processed-mask.txt',
    # 'poi-processed': 'det-poi-processed-mask.txt',
    'mrcnn-processed-mask': 'det-mrcnn-processed-mask.txt',
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def scale_bboxes_coord(bboxes, old_resolution, new_resolution):
    ratio = np.array(new_resolution) / np.array(old_resolution)
    x_scale = ratio[1]
    y_scale = ratio[0]
    bboxes[:, 0] *= x_scale
    bboxes[:, 1] *= y_scale
    bboxes[:, 2] *= x_scale
    bboxes[:, 3] *= y_scale
    return bboxes

def postprocessing(pred):
    # Predicted scores
    scores = pred['scores'].detach().cpu().numpy().tolist()
    lastid = [ scores.index(s) for s in scores if s > 0.3 ][-1]
    # Extract data
    masks = (pred['masks'] > 0.5).squeeze().detach().cpu().numpy()
    classes = [ COCO_INSTANCE_CATEGORY_NAMES[i]
                for i in pred['labels'].detach().cpu().numpy().tolist() ]
    bboxes  = [ (bbox[0], bbox[1], bbox[2], bbox[3])
                for bbox in pred['boxes'].detach().cpu().numpy().tolist() ]
    # Filter out data
    masks = masks[:lastid+1]
    bboxes = np.array(bboxes[:lastid+1])
    scores = np.array(scores[:lastid+1])
    classes = classes[:lastid+1]

    output_masks = []
    output_bboxes = []
    output_scores = []
    for bbox, score, cls, mask in zip(bboxes, scores, classes, masks):
        if cls != 'person':
            continue
        # Crop mask
        xmin, ymin = bbox[0], bbox[1]
        xmax, ymax = bbox[2], bbox[3]
        crop_mask = mask[int(ymin):int(ymax), int(xmin):int(xmax)]
        crop_mask = crop_mask.astype(np.uint8)

        output_masks.append(crop_mask)
        output_bboxes.append([ xmin+1, ymin+1, xmax-xmin, ymax-ymin ])
        output_scores.append(score)

    return output_bboxes, output_scores, output_masks

def tlwh_to_tlbr(tlwh):
    # Type checking
    if isinstance(tlwh, np.ndarray):
        tlwh = tlwh.tolist()
    elif isinstance(tlwh, list) or isinstance(tlwh, tuple):
        tlwh = tlwh
    else:
        raise Exception("Cannot handle data of type {}".format(type(tlwh)))

    # Conversion
    tl_x, tl_y, w, h = tuple(tlwh)
    return tl_x, tl_y, tl_x+w, tl_y+h

def iou(box1, box2):
    """Compute IoU between two bbox sets
    Arguments:
        box1 (tensor): tensor of shape (N, 4)
        box2 (tensor): tensor of shape (M, 4)
    Returns:
        tensor of shape (N, M) representing pair-by-pair iou values
        between two bbox sets.
    NOTES: box format (x1, y1, x2, y2)
    """
    epsilon = 1e-16
    N = box1.size(0)
    M = box2.size(0)
    # Compute intersection area
    lt = torch.max(
            box1[..., :2].unsqueeze(1).expand(N, M, 2), # (N, 2) -> (N, M, 2)
            box2[..., :2].unsqueeze(0).expand(N, M, 2), # (M, 2) -> (N, M, 2)
            )
    rb = torch.min(
            box1[..., 2:].unsqueeze(1).expand(N, M, 2), # (N, 2) -> (N, M, 2)
            box2[..., 2:].unsqueeze(0).expand(N, M, 2), # (M, 2) -> (N, M, 2)
            )
    wh = rb - lt                    # (N, M, 2)
    wh[wh<0] = 0                    # Non-overlapping conditions
    inter = wh[..., 0] * wh[..., 1] # (N, M)
    # Compute respective areas of boxes
    area1 = (box1[..., 2]-box1[..., 0]) * (box1[..., 3]-box1[..., 1]) # (N,)
    area2 = (box2[..., 2]-box2[..., 0]) * (box2[..., 3]-box2[..., 1]) # (M,)
    area1 = area1.unsqueeze(1).expand(N,M) # (N, M)
    area2 = area2.unsqueeze(0).expand(N,M) # (N, M)
    # Compute IoU
    iou = inter / (area1+area2-inter+epsilon)
    return iou.clamp(0).numpy()

def box_matching(bboxes, mboxes, threshold=0.3):
    # Convert box format tlwh to tlbr
    btlbrs = np.array([ tlwh_to_tlbr(tlwh) for tlwh in bboxes ])
    mtlbrs = np.array([ tlwh_to_tlbr(tlwh) for tlwh in mboxes ])

    cost_mat = 1-iou(torch.tensor(btlbrs), torch.tensor(mtlbrs))
    bindices, mindices = linear_sum_assignment(cost_mat)
    pairs = [ pair
            for pair in zip(bindices, mindices)
            if cost_mat[pair[0], pair[1]] <= threshold ]

    return pairs

def main(args):
    # MaskRCNN data preprocessing
    inverse = T.ToPILImage()
    mask_transform = T.Compose([ T.ToTensor() ])
    reid_transform = T.Compose([
                        T.Resize((MOTreID.IMG_HEIGHT, MOTreID.IMG_WIDTH)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                        ])

    # MaskRCNN Model
    idx = GPUtil.getFirstAvailable(order='memory')[0]
    device = f"cuda:{idx}"
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)
    model.to(device)
    model.eval()

    # ReID Model
    checkpoint = torch.load(args['reid'])
    reid = resnet50_reid(features=128, classes=1)
    reid.load_state_dict(checkpoint['model'])
    reid.to(device)
    reid.eval()

    # Process each detector
    for detector, detfile in LOOKUP.items():
        detfile = osp.join(args['sequence'], 'det', detfile)
        mask_dir = osp.join(args['sequence'], 'det', osp.basename(detfile).split('.')[0])
        if not osp.exists(mask_dir):
            os.makedirs(mask_dir)

        # sequence = MOTSequence(root=args['sequence'], mode='test', detector=detector)
        sequence = MOTSequence(root=args['sequence'], mode='test', detector='default-processed')
        print(sequence)

        # Add Mask & ReID
        result = {}
        mask_count = 0
        for idx in tqdm(range(len(sequence))):
            img, tboxes, bboxes, masks = sequence[idx]

            # Predicion objects & masks
            x = mask_transform(inverse(img)).to(device)
            pred = model([x])[0]
            mboxes, mscores, masks = postprocessing(pred)

            # Perfrom Assignment with IoU matrix
            if len(mboxes) == 0:
                continue
            # if len(bboxes) == 0 or len(mboxes) == 0:
                # continue
            # bboxes = np.array(bboxes)[:, 1:1+4]
            # pairs = box_matching(bboxes, mboxes, threshold=0.4)

            # Compute reid with masked objects
            eboxes = []
            emasks = []
            embeds = []
            escores = []
            for mbox, mscore, mask in zip(mboxes, mscores, masks):
            # for pair in pairs:
                # mbox = mboxes[pair[1]]
                # mask = masks[pair[1]]
                # mscore = mscores[pair[1]]

                # Compute reid
                xmin = int(mbox[0])
                ymin = int(mbox[1])
                xmax = xmin + int(mbox[2])
                ymax = ymin + int(mbox[3])
                try:
                    crop = img[:, ymin:ymax, xmin:xmax]
                    crop = inverse(crop)
                    crop = reid_transform(crop)
                except Exception as e:
                    continue
                crop = crop.to(device)
                embed = reid(crop.unsqueeze(0))[0]
                embed = embed.detach().cpu().numpy().tolist()

                eboxes.append(mbox)
                emasks.append(mask)
                embeds.append(embed)
                escores.append(mscore)

            # Save data in result
            frameId = idx+1
            if frameId not in result:
                result[frameId] = []
            for box, score, mask, embed in zip(eboxes, escores, emasks, embeds):
                mask_name = osp.join(mask_dir, f"mask{mask_count}.png")
                visMask = (mask * 255).astype("uint8")
                cv2.imwrite(mask_name, visMask)
                mask_count += 1

                relative_name = osp.join(osp.basename(mask_dir), osp.basename(mask_name))
                record = [ frameId ] + [-1] + box + [score, -1, -1, -1] + embed + [ relative_name ]
                result[frameId].append(record)

        # Save ReID result
        print(f"Export result to {detfile}")
        with open(detfile, 'w') as f:
            lines = []
            frameIds = sorted(result.keys())
            for frameId in frameIds:
                records = result[frameId]
                for record in records:
                    line = ",".join([ str(f) for f in record ])
                    lines.append(line)
            content = "\n".join(lines)
            f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", help="MOT Sequence")
    parser.add_argument("--reid", help="pretrained reid model")
    args = vars(parser.parse_args())
    main(args)

