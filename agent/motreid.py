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

from data.dataset.motreid import MOTreID, MOTreIDWrapper
from model.resnet import resnet50_reid
from loss.triplet import OnlineTripletLoss
from loss.utils import RandomNegativeTripletSelector


__all__ = [ "MOTreIDAgent" ]

class MOTreIDAgent:
    """Train Resnet50 with triplet loss"""
    def __init__(self, config):
        # Torch environment
        self.config = config
        device = config['train']['device'] if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Train Dataset (Batch size = P*K)
        self.dataset = MOTreIDWrapper(root=config['dataset']['root'],
                                    P=config['dataset']['P'],
                                    K=config['dataset']['K'],
                                    detector="frcnn", mode="train")
        self.dataset.use_transform(mode='train')

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
                                    steps_per_epoch=len(self.dataset))

        # Tensorboard
        self.logdir = config['train']['logdir']
        self.board = SummaryWriter(logdir=self.logdir)
        self.current_epoch = -1
        self.current_loss = 10000
        self.current_count = 10000

    def resume(self):
        checkpoint_path = osp.join(self.logdir, 'best.pth')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.schedular.load_state_dict(checkpoint['schedular'])
        self.current_loss = checkpoint['current_loss']
        self.current_count = checkpoint['current_count']
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
        loop = tqdm(list(range(len(self.dataset))),
                    leave=True,
                    desc=f"Train Epoch:{current_epoch}/{n_epochs}")
        triplet_losses = []
        triplet_counts = []
        self.model.train()
        for idx in loop:
            idx = np.random.randint(0, len(self.dataset))
            # Move data
            imgs, labels = self.dataset[idx]
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            # Forward & Backward
            embeddings, _ = self.model(imgs)
            self.optimizer.zero_grad()
            loss, count = self.criterion(embeddings, labels)
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
            self.board.add_scalar("Train Step Count", count, current_epoch*len(self.dataset)+idx)
            self.board.add_scalar("Train Step Loss", loss.item(), current_epoch*len(self.dataset)+idx)

        epoch_loss = sum(triplet_losses)/len(triplet_losses)
        epoch_count = sum(triplet_counts)/len(triplet_counts)
        if (
            epoch_loss < self.current_loss
            and epoch_count < self.current_count
        ):
            self.current_loss = epoch_loss
            self.current_count = epoch_count
            self._save_checkpoint()

    def validate(self):
        pass

    def finalize(self):
        pass

    def _save_checkpoint(self):
        checkpoint = {
            'model': self.model.eval().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'schedular': self.schedular.state_dict(),
            'current_loss': self.current_loss,
            'current_count': self.current_count,
            'current_epoch': self.current_epoch,
            }
        checkpoint_path = osp.join(self.logdir, 'best.pth')
        torch.save(checkpoint, checkpoint_path)
        print("Save checkpoint to '{}'".format(checkpoint_path))
