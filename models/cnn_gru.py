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


class CNNGRUModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.cnn = CNNEncoder(in_channels)

        self.gru = nn.GRU(
            input_size=128 + 2,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x_env, x_motion):
        B, T, C, H, W = x_env.shape

        feats = []
        for t in range(T):
            f = self.cnn(x_env[:, t])
            feats.append(f)

        feats = torch.stack(feats, dim=1)
        x = torch.cat([feats, x_motion], dim=2)

        _, h = self.gru(x)
        return self.fc(h[-1])
