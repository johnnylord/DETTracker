import numpy as np

import torch
import torchvision.transforms as T

from .motsequence import MOTSequence


class MOTreID(MOTSequence):
    IMG_WIDTH = 128
    IMG_HEIGHT = 256

    def __init__(
            self,
            P=16,
            K=4,
            min_crops_per_person=8,
            transform=None,
            **kwargs):
        super().__init__(**kwargs)
        self.P = P
        self.K = K
        self.min_crops_per_person = min_crops_per_person
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize((int(MOTreID.IMG_HEIGHT*1.125), int(MOTreID.IMG_WIDTH*1.125))),
                T.CenterCrop((MOTreID.IMG_HEIGHT, MOTreID.IMG_WIDTH)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ])
        self.data = self._build_people_profile()
        print(f"{len(self.data)} number of people are collected")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Generate Triplet sample from data"""
        # Select P candidates people
        targets = [idx]
        others = [ idx for idx in range(len(self.data)) if idx != targets[0] ]
        others = np.random.choice(others, self.P-1, replace=False).tolist()
        targets.extend(others)

        # Aggregate image data
        imgs = []
        labels = []
        for idx in targets:
            crops = self.data[idx]
            indices = torch.randperm(crops.size(0))
            crops = crops[indices[:self.K]]
            imgs.append(crops)
            labels.extend([idx]*self.K)

        imgs = torch.cat(imgs)
        labels = torch.LongTensor(labels)

        return imgs, labels

    def _build_people_profile(self):
        profiles = {}

        # Extract people profile frame-by-frame
        inverse = T.ToPILImage()
        seq_length = super().__len__()
        for idx in range(seq_length):
            print(f"Build Profile {idx}/{seq_length}", end="\r\b")
            img, tboxes, bboxes = super().__getitem__(idx)

            for box in tboxes:
                tid = int(box[0])
                xmin = int(box[1])
                ymin = int(box[2])
                xmax = int(box[1]+box[3])
                ymax = int(box[2]+box[4])
                if (xmax-xmin)*(ymax-ymin) == 0:
                    continue
                try:
                    crop = img[:, ymin:ymax, xmin:xmax]
                    crop = inverse(crop)
                    crop = self.transform(crop)
                except Exception as e:
                    continue

                if tid not in profiles:
                    profiles[tid] = []

                profiles[tid].append(crop)

        # Filter out profile with unsatisfied number of crops
        invalid_keys = [ k for k, v in profiles.items() if len(v) < self.min_crops_per_person ]
        for k in invalid_keys:
            del profiles[k]

        # Aggregate profile cropped image of each person
        data = []
        for crops in profiles.values():
            sample = torch.stack(crops, 0)
            data.append(sample)

        return data


if __name__ == "__main__":
    dataset = MOTreID(root="/home/johnnylord/dataset/MOT16/train/MOT16-02", detector='frcnn', mode='train')
    for i in range(len(dataset)):
        imgs, labels = dataset[i]
        print(imgs.shape, labels.shape)
