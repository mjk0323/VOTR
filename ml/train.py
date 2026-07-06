"""Training loop for the pitch estimator + technique classifier.

Run from the ml/ directory (needs data/, models/ importable at top level):
    python train.py --train-dir data/extracted/training --val-dir data/extracted/validation

For a quick CPU smoke test against the small sample from extract.py:
    python train.py --train-dir data/extracted/validation --epochs 1 --max-steps 3 --batch-size 8

Full training (many epochs, full dataset) is meant to run on a desktop GPU -
see docs/ml_pipeline.md for the GPU-vendor-specific setup.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import FileClusteredSampler, SingingNoteDataset
from models.pitch_estimator import PitchEstimator
from models.technique_classifier import TechniqueClassifier

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml  # AMD/Intel GPUs on Windows - see docs/ml_pipeline.md

        return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


def train(
    train_dir: Path,
    val_dir: Path | None,
    epochs: int,
    batch_size: int,
    lr: float,
    max_steps: int | None,
    num_workers: int,
) -> None:
    device = get_device()
    print(f"training on device: {device}")

    train_ds = SingingNoteDataset(train_dir)
    print(f"train notes: {len(train_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=FileClusteredSampler(train_ds),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    if val_dir is not None and val_dir.exists():
        val_ds = SingingNoteDataset(val_dir)
        print(f"val notes: {len(val_ds)}")

    pitch_model = PitchEstimator().to(device)
    technique_model = TechniqueClassifier().to(device)

    params = list(pitch_model.parameters()) + list(technique_model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    pitch_loss_fn = torch.nn.MSELoss()
    technique_loss_fn = torch.nn.BCEWithLogitsLoss()

    step = 0
    for epoch in range(epochs):
        pitch_model.train()
        technique_model.train()
        running_pitch_loss = 0.0
        running_technique_loss = 0.0
        n_batches = 0

        for mel, midi_target, technique_target in train_loader:
            mel = mel.to(device)
            midi_target = midi_target.to(device)
            technique_target = technique_target.to(device)

            optimizer.zero_grad()
            pitch_pred = pitch_model(mel)
            technique_logits = technique_model(mel)

            pitch_loss = pitch_loss_fn(pitch_pred, midi_target)
            technique_loss = technique_loss_fn(technique_logits, technique_target)
            loss = pitch_loss + technique_loss
            loss.backward()
            optimizer.step()

            running_pitch_loss += pitch_loss.item()
            running_technique_loss += technique_loss.item()
            n_batches += 1
            step += 1

            if max_steps is not None and step >= max_steps:
                break

        avg_pitch = running_pitch_loss / max(1, n_batches)
        avg_technique = running_technique_loss / max(1, n_batches)
        print(f"epoch {epoch + 1}/{epochs} - pitch_mse={avg_pitch:.3f} technique_bce={avg_technique:.3f}")

        if max_steps is not None and step >= max_steps:
            break

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(pitch_model.state_dict(), CHECKPOINT_DIR / "pitch_estimator.pt")
    torch.save(technique_model.state_dict(), CHECKPOINT_DIR / "technique_classifier.pt")
    print(f"checkpoints saved to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("data/extracted/training"))
    parser.add_argument("--val-dir", type=Path, default=Path("data/extracted/validation"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N steps (for smoke tests)")
    parser.add_argument("--num-workers", type=int, default=6, help="DataLoader worker processes")
    args = parser.parse_args()

    train(args.train_dir, args.val_dir, args.epochs, args.batch_size, args.lr, args.max_steps, args.num_workers)
