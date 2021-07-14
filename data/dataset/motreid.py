import os
import os.path as osp
import multiprocessing
import numpy as np

import torch
from torch.utils.data import ConcatDataset
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
                T.RandomCrop((MOTreID.IMG_HEIGHT, MOTreID.IMG_WIDTH)),
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
        # Apply training transformation
        inverse = T.ToPILImage()
        crops = []
        for img in imgs:
            img = inverse(img)
            crops.append(self.transform(img))
        crops = torch.stack(crops)

        return crops, labels

    def _build_people_profile(self):
        profiles = {}

        # Extract people profile frame-by-frame
        inverse = T.ToPILImage()
        preprocess = T.Compose([
                        T.Resize((MOTreID.IMG_HEIGHT, MOTreID.IMG_WIDTH)),
                        T.ToTensor()
                        ])
        seq_length = super().__len__()
        for idx in range(seq_length):
            print(f"Build Profile {idx}/{seq_length}", end="\r\b")
            img, tboxes, bboxes, masks = super().__getitem__(idx)

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
                    crop = preprocess(crop)
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


class MOTreIDWorker(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                self.task_queue.task_done()
                break
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return


class MOTreIDLoader:

    def __init__(self, root, P, K, min_crops_per_person, transform, **kwargs):
        self.root = root
        self.P = P
        self.K = K
        self.min_crops_per_person = min_crops_per_person
        self.transform = transform
        self.kwargs = kwargs

    def __call__(self):
        return MOTreID(
                    root=self.root,
                    P=self.P,
                    K=self.K,
                    transform=self.transform,
                    min_crops_per_person=self.min_crops_per_person,
                    mode='train', **self.kwargs)


class MOTreIDWrapper:
    CACHE_DIR = osp.join(osp.expanduser("~"), ".cache/torch/checkpoints")

    def __init__(
            self,
            # Wrapper Parameters
            root,
            num_workers=4,
            # MOTreID Parameters
            P=16, K=4,
            min_crops_per_person=8,
            transform=None,
            no_cache=False,
            # MOTSequence Parameters
            **kwargs):
        # Load Preprocessed dataset
        if not osp.exists(MOTreIDWrapper.CACHE_DIR):
            os.makedirs(MOTreIDWrapper.CACHE_DIR)
        if osp.exists(osp.join(MOTreIDWrapper.CACHE_DIR, 'motreids.pth')) and not no_cache:
            self.data = torch.load(osp.join(MOTreIDWrapper.CACHE_DIR, 'motreids.pth'))
        else:
            sequence_dirs = [ osp.join(root, seq) for seq in os.listdir(root) ]
            # Shared Queue between processes
            tasks = multiprocessing.JoinableQueue()
            results = multiprocessing.Manager().Queue()
            # Spawn reid loaders
            workers = [ MOTreIDWorker(tasks, results) for i in range(num_workers) ]
            for w in workers:
                w.start()
            # Put tasks in to queues
            for sequence_dir in sequence_dirs:
                loader = MOTreIDLoader(
                            root=sequence_dir,
                            P=P, K=K,
                            min_crops_per_person=min_crops_per_person,
                            transform=transform,
                            **kwargs)
                tasks.put(loader)
            # Put poison pill for each worker
            for i in range(num_workers):
                tasks.put(None)
            # Wait for all of the tasks to finish
            tasks.join()

            datasets = []
            while len(datasets) != len(sequence_dirs):
                dataset = results.get()
                datasets.append(dataset)

            self.data = ConcatDataset(datasets)
            checkpoint_path = osp.join(MOTreIDWrapper.CACHE_DIR, 'motreids.pth')
            torch.save(self.data, checkpoint_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def use_transform(self, mode='train'):
        if mode == 'train':
            transform = T.Compose([
                T.Resize((int(MOTreID.IMG_HEIGHT*1.125), int(MOTreID.IMG_WIDTH*1.125))),
                T.RandomCrop((MOTreID.IMG_HEIGHT, MOTreID.IMG_WIDTH)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ])
        elif mode == 'test':
            transform = T.Compose([
                T.Resize((MOTreID.IMG_HEIGHT, MOTreID.IMG_WIDTH)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ])
        elif mode == 'view':
            transform = T.Compose([
                T.Resize((MOTreID.IMG_HEIGHT, MOTreID.IMG_WIDTH)),
                T.ToTensor(),
                ])

        for dataset in self.data.datasets:
            dataset.transform = transform

    def use_PK(self, P, K):
        for dataset in self.data.datasets:
            dataset.P = P
            dataset.K = K


if __name__ == "__main__":
    # dataset = MOTreID(root="/home/johnnylord/dataset/MOT16/train/MOT16-02", detector='frcnn', mode='train')
    # for i in range(len(dataset)):
        # imgs, labels = dataset[i]
        # print(imgs.shape, labels.shape)

    dataset = MOTreIDWrapper(root="/home/johnnylord/dataset/MOT16/train/", detector='frcnn', num_workers=2, no_cache=True)
    print(len(dataset))
