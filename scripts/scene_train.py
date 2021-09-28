from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from model.scene import SceneDetector

DEVICE = "cuda"
DATA_DIR = "download/scene_rgb/"
LR_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 64

# Read dataset
transform = T.Compose([
                # T.Grayscale(),
                T.Resize((224, 224)),
                T.ToTensor(),
                # T.Normalize(mean=[0.5], std=[0.5]),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
dataset = ImageFolder(DATA_DIR, transform=transform)

# Split dataset
train_samples = int(len(dataset)*0.8)
valid_samples = len(dataset)-train_samples
train_dataset, valid_dataset = random_split(dataset, [train_samples, valid_samples])

# Create Dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)

# Create model
model = SceneDetector()
model = model.to(DEVICE)

# Create loss functino
criterion = nn.CrossEntropyLoss()

# Create optimizer
optimizer = optim.Adam(params=model.parameters(), lr=LR_RATE)

best_acc = 0.0
# Start training
for epoch in range(EPOCHS):

    accs = []
    losses = []
    model.train()
    for imgs, labels in tqdm(train_loader):
        # Move data
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        # Training
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Compute accuracy
        preds = torch.max(outputs.data, 1)[1]
        corrects = float(torch.sum(preds == labels.data))
        acc = corrects/len(imgs)
        # Compute loss values
        accs.append(acc)
        losses.append(loss.item())

    print(f"Train EPOCH{epoch}=> loss:{sum(losses)/len(losses)}, acc:{sum(accs)/len(accs)}")

    accs = []
    losses = []
    model.eval()
    for imgs, labels in tqdm(valid_loader):
        # Move data
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        # Training
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        # Compute accuracy
        preds = torch.max(outputs.data, 1)[1]
        corrects = float(torch.sum(preds == labels.data))
        acc = corrects/len(imgs)
        # Compute loss values
        accs.append(acc)
        losses.append(loss.item())

    mean_acc = sum(accs)/len(accs)
    if mean_acc > best_acc:
        best_acc = mean_acc
        torch.save(model, 'scene.pth')
        print("Save model")

    print(f"Valid EPOCH{epoch}=> loss:{sum(losses)/len(losses)}, acc:{sum(accs)/len(accs)}")
