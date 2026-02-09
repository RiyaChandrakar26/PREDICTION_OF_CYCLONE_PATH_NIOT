import torch
import torch.nn as nn


class MotionGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=2,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        _, h = self.gru(x)
        h = h[-1]
        return self.fc(h)
