"""Runs the Track B (ml/) trained pitch/technique ONNX models against a full
song: segments the waveform into per-note windows (the models were trained
on individually-cropped notes, not continuous audio - see
ml/data/dataset.py), then runs each segment through both models.

The log-mel computation here must numerically match ml/data/dataset.py's
(which uses torchaudio.transforms) - verified to match to within ~1.6e-4 dB
using librosa.feature.melspectrogram with matching parameters (htk mel scale,
no filterbank norm, no top_db clamp). Backend intentionally avoids a torch
dependency (see requirements.txt / report_generation.py's Groq-over-Ollama
choice for the same reason) - onnxruntime + librosa is enough here.

Decision thresholds (bending/vibrato/breath) come from ml/evaluate.py's
per-label threshold sweep on the validation split - not the default 0.5,
which was measurably worse (see ml/docs/ml_pipeline.md history).
"""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort

MODEL_DIR = Path(__file__).resolve().parents[1] / "ml_models"

TARGET_SR = 22050
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
MAX_FRAMES = 87  # ~1s at hop_length=256, sr=22050 - matches training
MIN_NOTE_DURATION_SEC = 0.05

BENDING_THRESHOLD = 0.66
VIBRATO_THRESHOLD = 0.62
BREATH_THRESHOLD = 0.70


@dataclass
class NoteResult:
    start_time: float
    end_time: float
    cents_deviation: float  # predicted pitch minus nearest semitone, in cents
    is_bending: bool
    is_vibrato: bool
    is_breath: bool


_pitch_session: ort.InferenceSession | None = None
_technique_session: ort.InferenceSession | None = None


def _sessions() -> tuple[ort.InferenceSession, ort.InferenceSession]:
    global _pitch_session, _technique_session
    if _pitch_session is None:
        _pitch_session = ort.InferenceSession(str(MODEL_DIR / "pitch_estimator.onnx"))
        _technique_session = ort.InferenceSession(str(MODEL_DIR / "technique_classifier.onnx"))
    return _pitch_session, _technique_session


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y.astype(np.float32)
    duration = len(y) / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0, 1, num=len(y), endpoint=False)
    x_new = np.linspace(0, 1, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, y).astype(np.float32)


def _to_log_mel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        n_mels=N_MELS,
        fmin=0.0,
        fmax=TARGET_SR / 2,
        htk=True,
        norm=None,
    )
    log_mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)

    frames = log_mel.shape[-1]
    if frames < MAX_FRAMES:
        pad = MAX_FRAMES - frames
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant", constant_values=float(log_mel.min()))
    else:
        log_mel = log_mel[:, :MAX_FRAMES]
    return log_mel.astype(np.float32)


def _segment_notes(y: np.ndarray, sr: int) -> list[tuple[float, float]]:
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
    duration = len(y) / sr

    boundaries = sorted({0.0, *onset_times, duration})
    return [
        (start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end - start >= MIN_NOTE_DURATION_SEC
    ]


def analyze_notes(y: np.ndarray, sr: int) -> list[NoteResult]:
    segments = _segment_notes(y, sr)
    if not segments:
        return []

    mels = []
    for start, end in segments:
        seg_y = y[int(start * sr) : int(end * sr)]
        seg_y = _resample(seg_y, sr, TARGET_SR)
        mels.append(_to_log_mel(seg_y))
    batch = np.stack(mels)[:, np.newaxis, :, :].astype(np.float32)  # (N, 1, n_mels, frames)

    pitch_session, technique_session = _sessions()
    pitch_pred = pitch_session.run(None, {"mel": batch})[0]  # (N,) predicted MIDI numbers
    technique_probs = technique_session.run(None, {"mel": batch})[0]  # (N, 3): bending, vibrato, breath

    nearest = np.round(pitch_pred)
    cents_deviation = (pitch_pred - nearest) * 100

    return [
        NoteResult(
            start_time=start,
            end_time=end,
            cents_deviation=float(cents_deviation[i]),
            is_bending=bool(technique_probs[i, 0] > BENDING_THRESHOLD),
            is_vibrato=bool(technique_probs[i, 1] > VIBRATO_THRESHOLD),
            is_breath=bool(technique_probs[i, 2] > BREATH_THRESHOLD),
        )
        for i, (start, end) in enumerate(segments)
    ]
