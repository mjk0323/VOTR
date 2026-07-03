"""Singing-voice pitch estimator: log-mel spectrogram of one note -> MIDI pitch.

Trained on real ground-truth MIDI labels from the AI Hub dataset (instead of
being a heuristic like librosa.pyin), this is meant to eventually replace
analyze_pitch() in backend/app/services/audio_analysis.py.
"""

import torch.nn as nn

from models.backbone import SmallCNNBackbone


class PitchEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SmallCNNBackbone()
        self.head = nn.Linear(self.backbone.out_features, 1)

    def forward(self, x):
        # returns predicted MIDI pitch number, shape (batch,)
        return self.head(self.backbone(x)).squeeze(-1)
