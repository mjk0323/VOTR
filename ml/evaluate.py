"""Evaluate trained checkpoints against the Validation split.

Run from the ml/ directory:
    python evaluate.py --val-dir data/extracted/validation
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import SingingNoteDataset
from models.pitch_estimator import PitchEstimator, decode_pitch
from models.technique_classifier import LABELS, TechniqueClassifier
from train import get_device

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def evaluate(val_dir: Path, batch_size: int = 32) -> None:
    device = get_device()

    val_ds = SingingNoteDataset(val_dir)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    print(f"val notes: {len(val_ds)}")

    pitch_model = PitchEstimator().to(device)
    pitch_model.load_state_dict(torch.load(CHECKPOINT_DIR / "pitch_estimator.pt", map_location=device))
    pitch_model.eval()

    technique_model = TechniqueClassifier().to(device)
    technique_model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "technique_classifier.pt", map_location=device)
    )
    technique_model.eval()

    abs_errors_semitones: list[float] = []
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for mel, midi_target, technique_target in val_loader:
            mel = mel.to(device)
            midi_target = midi_target.to(device)
            technique_target = technique_target.to(device)

            pitch_logits = pitch_model(mel)
            pitch_pred = decode_pitch(pitch_logits)
            not_breath = technique_target[:, 2] < 0.5  # breath notes have no real pitch
            errors = (pitch_pred - midi_target).abs()
            abs_errors_semitones.extend(errors[not_breath].cpu().tolist())

            technique_logits = technique_model(mel)
            all_probs.append(torch.sigmoid(technique_logits).cpu())
            all_targets.append(technique_target.cpu())

    mean_abs_error_semitones = sum(abs_errors_semitones) / max(1, len(abs_errors_semitones))
    mean_abs_error_cents = mean_abs_error_semitones * 100
    print(f"pitch: mean abs error = {mean_abs_error_semitones:.3f} semitones ({mean_abs_error_cents:.1f} cents)")

    probs = torch.cat(all_probs)  # (N, len(LABELS))
    targets = torch.cat(all_targets)

    # Sweep the decision threshold per label and report the one that maximizes F1
    # (not raw accuracy - these labels are all <10% positive, so "predict never"
    # already scores >90% accuracy without being useful) rather than assuming 0.5.
    for i, label in enumerate(LABELS):
        probs_i = probs[:, i]
        targets_i = targets[:, i]
        best = (0.0, 0.5, 0.0, 0.0)  # f1, threshold, precision, recall
        for threshold in torch.linspace(0.05, 0.95, 91).tolist():
            pred_i = (probs_i > threshold).float()
            tp = ((pred_i == 1) & (targets_i == 1)).sum().item()
            fp = ((pred_i == 1) & (targets_i == 0)).sum().item()
            fn = ((pred_i == 0) & (targets_i == 1)).sum().item()
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-9, precision + recall)
            if f1 > best[0]:
                best = (f1, threshold, precision, recall)
        f1, threshold, precision, recall = best
        print(f"{label}: best_threshold={threshold:.2f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", type=Path, default=Path("data/extracted/validation"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    evaluate(args.val_dir, args.batch_size)
