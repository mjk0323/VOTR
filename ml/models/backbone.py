"""Shared small CNN feature extractor for note-length log-mel spectrograms.

Kept intentionally small (3 conv blocks, global average pool) so a CPU smoke
test (a few batches on a laptop) finishes in seconds - real training runs on
a desktop GPU per docs/ml_pipeline.md.
"""

import torch.nn as nn


class SmallCNNBackbone(nn.Module):
    def __init__(self, out_features: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_features = out_features

    def forward(self, x):
        # x: (batch, 1, n_mels, frames) -> (batch, out_features)
        return self.conv(x).flatten(1)
