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
    tp = [0] * len(LABELS)
    fp = [0] * len(LABELS)
    fn = [0] * len(LABELS)

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
            technique_pred = (torch.sigmoid(technique_logits) > 0.5).float()

            for i in range(len(LABELS)):
                pred_i = technique_pred[:, i]
                target_i = technique_target[:, i]
                tp[i] += int(((pred_i == 1) & (target_i == 1)).sum().item())
                fp[i] += int(((pred_i == 1) & (target_i == 0)).sum().item())
                fn[i] += int(((pred_i == 0) & (target_i == 1)).sum().item())

    mean_abs_error_semitones = sum(abs_errors_semitones) / max(1, len(abs_errors_semitones))
    mean_abs_error_cents = mean_abs_error_semitones * 100
    print(f"pitch: mean abs error = {mean_abs_error_semitones:.3f} semitones ({mean_abs_error_cents:.1f} cents)")

    for i, label in enumerate(LABELS):
        precision = tp[i] / max(1, tp[i] + fp[i])
        recall = tp[i] / max(1, tp[i] + fn[i])
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        print(f"{label}: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", type=Path, default=Path("data/extracted/validation"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    evaluate(args.val_dir, args.batch_size)
