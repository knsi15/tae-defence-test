import torch.nn as nn


class TSAutoencoder(nn.Module):
    def __init__(self, T, bottleneck=None):
        super().__init__()
        if bottleneck is None:
            bottleneck = max(8, T // 8)

        h1 = T // 2
        h2 = T // 4

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(T, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, bottleneck),
            nn.ReLU(),
        )
        # decoder
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, T),
        )

    def forward(self, x):
        # x: (B, T) を想定。(B, T, 1, 1) で来たら view で平坦化
        x = x.view(x.size(0), -1)
        z = self.enc(x)
        return self.dec(z)
