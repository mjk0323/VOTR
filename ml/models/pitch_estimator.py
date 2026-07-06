"""Singing-voice pitch estimator: log-mel spectrogram of one note -> MIDI pitch.

Trained on real ground-truth MIDI labels from the AI Hub dataset (instead of
being a heuristic like librosa.pyin), this is meant to eventually replace
analyze_pitch() in backend/app/services/audio_analysis.py.

Uses a CREPE-style pitch-bin classifier rather than scalar MIDI regression:
the head outputs logits over per-semitone bins spanning the dataset's
observed range, trained against a Gaussian-smoothed soft target (so nearby
bins still get partial credit) and decoded via an expected-value weighted
average for sub-bin precision. A first pass with plain MSE regression gave
~209 cents mean error - classification is what real pitch trackers (CREPE,
SPICE) use for exactly this reason.

Breath notes have no real pitch and are given a dummy placeholder MIDI value
in the dataset's own labels (see data/dataset.py) - `pitch_loss` takes a mask
so callers can exclude them from the loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import N_MELS
from models.backbone import FreqPreservingCNNBackbone

MIN_MIDI = 34
MAX_MIDI = 88
NUM_BINS = MAX_MIDI - MIN_MIDI + 1
BIN_SIGMA = 1.0  # semitones - width of the Gaussian soft target


class PitchEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FreqPreservingCNNBackbone(n_mels=N_MELS)
        self.head = nn.Linear(self.backbone.out_features, NUM_BINS)

    def forward(self, x):
        # returns per-semitone-bin logits, shape (batch, NUM_BINS)
        return self.head(self.backbone(x))


def _bin_centers(device: torch.device) -> torch.Tensor:
    return torch.arange(NUM_BINS, dtype=torch.float32, device=device) + MIN_MIDI


def midi_to_soft_target(midi_num: torch.Tensor) -> torch.Tensor:
    """(batch,) MIDI numbers -> (batch, NUM_BINS) Gaussian-smoothed soft labels."""
    centers = _bin_centers(midi_num.device)
    diff = midi_num.unsqueeze(1) - centers.unsqueeze(0)
    weights = torch.exp(-0.5 * (diff / BIN_SIGMA) ** 2)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)


def decode_pitch(logits: torch.Tensor) -> torch.Tensor:
    """(batch, NUM_BINS) logits -> (batch,) predicted MIDI number (expected value)."""
    centers = _bin_centers(logits.device)
    probs = torch.softmax(logits, dim=1)
    return (probs * centers.unsqueeze(0)).sum(dim=1)


def pitch_loss(logits: torch.Tensor, midi_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL divergence between the predicted and Gaussian-smoothed target bin
    distributions, averaged only over `mask` (1 = real note, 0 = excluded -
    e.g. a breath note with a dummy placeholder pitch)."""
    soft_target = midi_to_soft_target(midi_target)
    log_probs = F.log_softmax(logits, dim=1)
    kl_per_sample = F.kl_div(log_probs, soft_target, reduction="none").sum(dim=1)
    denom = mask.sum().clamp_min(1.0)
    return (kl_per_sample * mask).sum() / denom
