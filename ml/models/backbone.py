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


class FreqPreservingCNNBackbone(nn.Module):
    """Like SmallCNNBackbone, but only pools over time - the frequency axis
    stays intact all the way to the output. For pitch estimation, SmallCNNBackbone's
    full 2D global average pool throws away exactly the information that
    identifies the pitch: which frequency band is active."""

    def __init__(self, n_mels: int, out_channels: int = 48):
        super().__init__()
        freq_out = max(1, n_mels // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),  # halve time only - keep every frequency bin
            nn.Conv2d(24, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # halve both - frequency resolution is still 2x finer than a semitone bin
            nn.AdaptiveAvgPool2d((freq_out, 1)),  # collapse remaining time, keep the freq axis
        )
        self.out_features = out_channels * freq_out

    def forward(self, x):
        # x: (batch, 1, n_mels, frames) -> (batch, out_channels * n_mels)
        return self.conv(x).flatten(1)
