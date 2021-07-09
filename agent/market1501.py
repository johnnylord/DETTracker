import os
import os.path as osp
import random

import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.optim.lr_scheduler import OneCycleLR
from tensorboardX import  SummaryWriter

from torch.utils.data import DataLoader
from data.sampler import BalancedBatchSampler
from data.dataset.market1501 import Market1501
from model.resnet import resnet50_reid
from loss.triplet import OnlineTripletLoss
from loss.utils import RandomNegativeTripletSelector


__all__ = [ "Market1501Agent" ]

class Market1501Agent:
    """Train Resnet50 with triplet loss on market1501 dataset"""
    def __init__(self, config):
        # Torch environment
        self.config = config
        device = config['train']['device'] if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Train Dataset (Batch size = P*K)
        tr_transform = T.Compose([
            T.Resize(config['dataset']['size']),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop(config['dataset']['size']),
            T.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ])
        te_transform = T.Compose([
            T.Resize(config['dataset']['size']),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
            ])
        train_dataset = Market1501(root=config['dataset']['root'], transform=tr_transform, mode='all')
        query_dataset = Market1501(root=config['dataset']['root'], transform=te_transform, mode='query')
        gallery_dataset = Market1501(root=config['dataset']['root'], transform=te_transform, mode='gallery')

        print("Training Dataset")
        print(train_dataset)

        # Define dataloader
        common_config = { 'num_workers': 4, 'pin_memory': False }
        train_labels = [ sample[1] for sample in train_dataset.data ]
        sampler = BalancedBatchSampler(train_labels,
                                    P=config['dataloader']['sampler']['P'],
                                    K=config['dataloader']['sampler']['K'])
        self.train_loader = DataLoader(train_dataset, batch_sampler=sampler, **common_config)
        self.query_loader = DataLoader(query_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                shuffle=False, **common_config)
        self.gallery_loader = DataLoader(gallery_dataset,
                                batch_size=config['dataloader']['batch_size'],
                                shuffle=False, **common_config)

        # Construct Model
        model = resnet50_reid(features=config['model']['features'], classes=1)
        model = model.to(self.device)
        self.model = model.train()

        # Learning Objective
        margin = config['loss']['margin']
        selector = RandomNegativeTripletSelector(margin=margin)
        self.criterion = OnlineTripletLoss(margin, selector)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config['optimizer']['lr'],
                                    weight_decay=config['optimizer']['weight_decay'])
        self.schedular = OneCycleLR(self.optimizer,
                                    max_lr=config['optimizer']['lr'],
                                    epochs=config['train']['n_epochs'],
                                    steps_per_epoch=len(self.train_loader))

        # Tensorboard
        self.logdir = config['train']['logdir']
        self.board = SummaryWriter(logdir=self.logdir)
        self.current_epoch = -1
        self.current_map = 0.0

    def resume(self):
        checkpoint_path = osp.join(self.logdir, 'best.pth')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.schedular.load_state_dict(checkpoint['schedular'])
        self.current_map = checkpoint['current_map']
        self.current_epoch = checkpoint['current_epoch']
        print("Resume training at epoch '{}'".format(self.current_epoch))

    def train(self):
        for epoch in range(self.current_epoch+1, self.config['train']['n_epochs']):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.validate()

    def train_one_epoch(self):
        current_epoch = self.current_epoch
        n_epochs = self.config['train']['n_epochs']
        loop = tqdm(self.train_loader,
                    leave=True,
                    desc=f"Train Epoch:{current_epoch}/{n_epochs}")
        triplet_losses = []
        triplet_counts = []
        self.model.train()
        for batch_idx, (imgs, pids, _, _)  in enumerate(loop):
            # Move data
            imgs = imgs.to(self.device)
            pids = pids.to(self.device)
            # Forward & Backward
            embeddings, _ = self.model(imgs)
            self.optimizer.zero_grad()
            loss, count = self.criterion(embeddings, pids)
            loss.backward()
            self.optimizer.step()
            self.schedular.step()
            triplet_counts.append(count)
            triplet_losses.append(loss.item())
            loop.set_postfix(
                    lr=self.optimizer.param_groups[0]['lr'],
                    loss=sum(triplet_losses)/len(triplet_losses),
                    count=sum(triplet_counts)/len(triplet_counts))
            # Logging
            self.board.add_scalar("Train Step Count", count, current_epoch*len(self.train_loader)+batch_idx)
            self.board.add_scalar("Train Step Loss", loss.item(), current_epoch*len(self.train_loader)+batch_idx)

    def validate(self):
        current_epoch = self.current_epoch
        n_epochs = self.config['train']['n_epochs']
        self.model.eval()
        # Compute query embeddings
        loop = tqdm(self.query_loader,
                    leave=True,
                    desc=f"Query Embedding:{current_epoch}/{n_epochs}")
        query_embeds, query_pids, query_camids = [], [], []
        for imgs, pids, camids, _ in loop:
            # Copmute embeddings
            imgs = imgs.to(self.device)
            embeds = self.model(imgs)
            # Save result
            query_embeds.append(embeds.detach().cpu().numpy())
            query_camids.append(camids.detach().cpu().numpy())
            query_pids.append(pids.detach().cpu().numpy())
        # Aggregate queries
        query_embeds = np.concatenate(query_embeds)
        query_camids = np.concatenate(query_camids)
        query_pids = np.concatenate(query_pids)

        # Compute gallery embeddings
        loop = tqdm(self.gallery_loader,
                    leave=True,
                    desc=f"Gallery Embedding:{current_epoch}/{n_epochs}")
        gallery_embeds, gallery_pids, gallery_camids = [], [], []
        for imgs, pids, camids, _ in loop:
            # Copmute embeddings
            imgs = imgs.to(self.device)
            embeds = self.model(imgs)
            # Save result
            gallery_embeds.append(embeds.detach().cpu().numpy())
            gallery_camids.append(camids.detach().cpu().numpy())
            gallery_pids.append(pids.detach().cpu().numpy())
        # Aggregate queries
        gallery_embeds = np.concatenate(gallery_embeds)
        gallery_camids = np.concatenate(gallery_camids)
        gallery_pids = np.concatenate(gallery_pids)

        # Copmute mAP & CMC
        ap = 0.0
        cmc = np.zeros(len(gallery_embeds))

        # Evaluate query with gallery set one-by-one
        n_valid_evaluations = 0
        for query_embed, query_pid, query_camid in zip(query_embeds, query_pids, query_camids):
            # Compute cosine similarity score with gallerys
            scores = gallery_embeds.dot(query_embed.T)
            sorted_indices = np.argsort(scores)
            sorted_indices = sorted_indices[::-1]

            # Valid gallery images (Same pid but different camid)
            same_pid_indices = np.argwhere(gallery_pids == query_pid)
            same_cam_indices = np.argwhere(gallery_camids == query_camid)
            target_indices = np.setdiff1d(same_pid_indices, same_cam_indices, assume_unique=True)

            # Invalid gallery images (pid is background, pid and camid is same as query)
            junk1_indices = np.argwhere(gallery_pids == -1)
            junk2_indices = np.intersect1d(same_pid_indices, same_cam_indices)
            invalid_indices = np.append(junk1_indices, junk2_indices)

            # Compute ap and cmc
            ret, ap_tmp, cmc_tmp = compute_AP_CMC(sorted_indices, target_indices, invalid_indices)
            if ret:
                ap = ap + ap_tmp
                cmc = cmc + cmc_tmp
                n_valid_evaluations += 1

        mAP = ap / n_valid_evaluations
        CMC = cmc / n_valid_evaluations
        print((
            f"Epoch {self.current_epoch}:{self.config['train']['n_epochs']},"
            f" Rank@1: {CMC[0]},"
            f" Rank@5: {CMC[4]},"
            f" Randk@10: {CMC[9]},"
            f" mAP: {mAP}"
            ))

        if mAP > self.current_map:
            self.current_map = mAP
            self._save_checkpoint()

    def finalize(self):
        pass

    def _save_checkpoint(self):
        checkpoint = {
            'model': self.model.eval().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'schedular': self.schedular.state_dict(),
            'current_map': self.current_map,
            'current_epoch': self.current_epoch,
            }
        checkpoint_path = osp.join(self.logdir, 'best.pth')
        torch.save(checkpoint, checkpoint_path)
        print("Save checkpoint to '{}'".format(checkpoint_path))


def compute_AP_CMC(indices, target_indices, invalid_indices):
    """Compute ap(average precision) and CMC(cumulative matching characteristic)
    Args:
        indices (list): list of ranked index
        target_indices (list): list of target index
        invalid_indices (list): list of invalid index
    """
    ap = 0.0
    cmc = np.zeros(len(indices))

    if not len(target_indices):
        return False, ap, cmc

    # Remove invalid indices from indices
    mask = np.in1d(indices, invalid_indices, invert=True)
    indices = indices[mask]

    # Find matching index
    mask = np.in1d(indices, target_indices)
    match_indices = np.argwhere(mask==True)
    match_indices = match_indices.flatten()

    # Copmute ap and cmc
    cmc[match_indices[0]:] = 1.

    n_targets = len(target_indices)
    for i in range(1, n_targets+1):
        d_recall = 1. / n_targets
        precision = float(i) / (match_indices[i-1]+1)

        if match_indices[i-1] != 0:
            old_precision = float(i-1) / match_indices[i-1]
        else:
            old_precision = 1.0

        ap = ap + d_recall*(old_precision+precision)/2

    return True, ap, cmc
