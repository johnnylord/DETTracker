import os
import os.path as osp
import argparse

from tqdm import tqdm
import GPUtil
import torch
import torchvision.transforms as T
from data.dataset.motreid import MOTreID
from data.dataset.motsequence import MOTSequence
from model.resnet import resnet50_reid

LOOKUP = {
    # Default detector with MOTreID feature
    'default-processed': 'default-processed.txt',
    'frcnn-processed': 'frcnn-processed.txt',
    'poi-processed': 'poi-processed.txt',
}
def main(args):
    # ReID data preprocessing
    inverse = T.ToPILImage()
    transform = T.Compose([
        T.Resize((MOTreID.IMG_HEIGHT, MOTreID.IMG_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ])
    # ReID Model
    idx = GPUtil.getFirstAvailable(order='memory')[0]
    device = f"cuda:{idx}"
    checkpoint = torch.load(args['reid'])
    model = resnet50_reid(features=128, classes=1)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    detectors = list(LOOKUP.keys())
    for detector in detectors:
        try:
            sequence = MOTSequence(root=args['sequence'], mode='test', detector=detector)
        except Exception as e:
            continue

        # Add ReID
        print(f"Process sequence with detector {detector}")
        result = {}
        for idx in tqdm(range(len(sequence))):
            img, tboxes, bboxes, masks = sequence[idx]
            # Filter out bboxes with small area
            bboxes = [ box for box in bboxes if int(box[3])*int(box[4]) > 0 ]
            # Compute box embedding
            eboxes = []
            embeds = []
            for box in bboxes:
                xmin = int(box[1])
                ymin = int(box[2])
                xmax = xmin + int(box[3])
                ymax = ymin + int(box[4])
                try:
                    crop = img[:, ymin:ymax, xmin:xmax]
                    crop = inverse(crop)
                    crop = transform(crop)
                except Exception as e:
                    continue
                crop = crop.to(device)
                embed = model(crop.unsqueeze(0))[0]
                embed = embed.detach().cpu().numpy().tolist()
                embeds.append(embed)
                eboxes.append(box)
            # Save data in result
            frameId = idx+1
            if frameId not in result:
                result[frameId] = []
            for box, embed in zip(eboxes, embeds):
                record = [ frameId ] + box + [-1, -1, -1] + embed
                result[frameId].append(record)

        # Save ReID result
        output = osp.join(args['sequence'], 'det', LOOKUP[detector])
        print(f"Export result to {output}")
        with open(output, 'w') as f:
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
    parser.add_argument("--reid", help="pretrained reid checkpoint")
    args = vars(parser.parse_args())
    main(args)
