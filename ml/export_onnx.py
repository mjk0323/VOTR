"""Exports the trained checkpoints to ONNX for inference in the backend
(which uses onnxruntime, not torch - see docs/ml_pipeline.md).

Both wrappers below bake the post-processing into the exported graph (bin
decoding for pitch, sigmoid for technique) so the backend only has to run the
model and read off final values - it doesn't need to reimplement
decode_pitch's bin-center math or know MIN_MIDI/NUM_BINS.

Run from the ml/ directory:
    python export_onnx.py
"""

from pathlib import Path

import torch
import torch.nn as nn

from models.pitch_estimator import PitchEstimator, decode_pitch
from models.technique_classifier import TechniqueClassifier

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


class _PitchExportWrapper(nn.Module):
    def __init__(self, model: PitchEstimator):
        super().__init__()
        self.model = model

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return decode_pitch(self.model(mel))  # (batch,) predicted MIDI number


class _TechniqueExportWrapper(nn.Module):
    def __init__(self, model: TechniqueClassifier):
        super().__init__()
        self.model = model

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(mel))  # (batch, 3) probabilities


def export() -> None:
    dummy = torch.zeros(1, 1, 80, 87, dtype=torch.float32)
    dynamic_axes = {"mel": {0: "batch"}, "output": {0: "batch"}}

    pitch_model = PitchEstimator()
    pitch_model.load_state_dict(torch.load(CHECKPOINT_DIR / "pitch_estimator.pt", map_location="cpu"))
    pitch_model.eval()
    torch.onnx.export(
        _PitchExportWrapper(pitch_model),
        dummy,
        str(CHECKPOINT_DIR / "pitch_estimator.onnx"),
        input_names=["mel"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )
    print(f"wrote {CHECKPOINT_DIR / 'pitch_estimator.onnx'}")

    technique_model = TechniqueClassifier()
    technique_model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "technique_classifier.pt", map_location="cpu")
    )
    technique_model.eval()
    torch.onnx.export(
        _TechniqueExportWrapper(technique_model),
        dummy,
        str(CHECKPOINT_DIR / "technique_classifier.onnx"),
        input_names=["mel"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )
    print(f"wrote {CHECKPOINT_DIR / 'technique_classifier.onnx'}")


if __name__ == "__main__":
    export()
