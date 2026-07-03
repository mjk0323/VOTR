"""Vocal technique multi-label classifier: log-mel spectrogram of one note ->
[is_bending, is_vibrt, is_breath] logits.

Trained on the AI Hub dataset's per-note technique flags. Meant to feed richer,
label-grounded "노래 스킬"/표현력 commentary into report_generation.py instead of
the current DSP-heuristic vibrato/bend detection.
"""

import torch.nn as nn

from models.backbone import SmallCNNBackbone

LABELS = ["is_bending", "is_vibrt", "is_breath"]


class TechniqueClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SmallCNNBackbone()
        self.head = nn.Linear(self.backbone.out_features, len(LABELS))

    def forward(self, x):
        # returns raw logits, shape (batch, 3) - use BCEWithLogitsLoss / sigmoid
        return self.head(self.backbone(x))
