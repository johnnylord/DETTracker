import torch
import torch.nn as nn


class SceneDetector(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
                            # Conv1 (1, 224, 224)
                            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(16),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2),
                            # Conv2 (16, 112, 112)
                            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2),
                            # Conv3 (32, 56, 56)
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2),
                            # Conv4 (64, 28, 28)
                            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2),
                            # Conv5 (128, 14, 14)
                            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2, 2),
                            # (256, 7, 7)
                            )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
                            nn.Linear(256, 64),
                            nn.ReLU(inplace=True),
                            nn.Linear(64, 2),
                            )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(batch_size, -1)
        out = self.classifier(x)
        return out


if __name__ == "__main__":
    x = torch.rand(1, 3, 224, 224)

    model = SceneDetector()
    y = model(x)
    print(y)
