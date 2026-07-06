"""Vocal technique multi-label classifier: log-mel spectrogram of one note ->
[is_bending, is_vibrt, is_breath] logits.

Trained on the AI Hub dataset's per-note technique flags. Meant to feed richer,
label-grounded "노래 스킬"/표현력 commentary into report_generation.py instead of
the current DSP-heuristic vibrato/bend detection.
"""

import torch
import torch.nn as nn

from models.backbone import TimePreservingCNNBackbone

LABELS = ["is_bending", "is_vibrt", "is_breath"]

# sqrt(negatives/positives) measured on the training split (is_bending 8.13%,
# is_vibrt 3.88%, is_breath 7.38% of all notes) - a dampened version of the
# standard BCEWithLogitsLoss pos_weight class-imbalance correction. The full
# undamped ratio (~11x/25x/13x) overcorrects and tanks precision; sqrt trades
# some of that recall gain back for precision.
POS_WEIGHT = torch.tensor([3.36, 4.98, 3.54])


class TechniqueClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TimePreservingCNNBackbone()
        self.head = nn.Linear(self.backbone.out_features, len(LABELS))

    def forward(self, x):
        # returns raw logits, shape (batch, 3) - use BCEWithLogitsLoss / sigmoid
        return self.head(self.backbone(x))
