import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)

class MotionGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=2,
            hidden_size=64,
            batch_first=True
        )

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]


class GRUFusionModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.cnn = CNNEncoder(in_channels)
        self.gru = MotionGRU()

        self.fc = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x_env, x_motion):
        f_env = self.cnn(x_env)
        f_motion = self.gru(x_motion)

        x = torch.cat([f_env, f_motion], dim=1)
        return self.fc(x)

