"""Note-level PyTorch Dataset built from the AI Hub "다음색 가이드보컬" WAV+JSON pairs.

Each item is a short audio segment (one sung note) paired with two supervised
targets sourced directly from the dataset's ground-truth labels:
  - midi_target: the note's true MIDI pitch number (for the pitch estimator)
  - technique_target: [is_bending, is_vibrt, is_breath] multi-hot vector
    (for the vocal technique classifier)

Audio I/O uses soundfile. Note: torchaudio's I/O functions (load/info) are
unavailable in this environment due to a torch/torchaudio version mismatch
(see docs/ml_pipeline.md) - only torchaudio.transforms (pure PyTorch, no C++
extension) is used here.

A song file has hundreds of notes on average, so re-opening and seeking into
it per note (one full song per file, but each note is a separate small read)
made per-note file I/O the dominant training cost. `_load_note_audio` instead
keeps a small per-process LRU cache of fully-decoded songs and slices notes
out of memory; `FileClusteredSampler` shuffles at the file level (not the raw
note level) so that consecutive notes fed to a DataLoader worker tend to share
a file, keeping that cache's hit rate high.
"""

import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset, Sampler

MIN_NOTE_DURATION_SEC = 0.05
TARGET_SR = 22050
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
MAX_FRAMES = 87  # ~1s of audio at hop_length=256, sr=22050 - covers most single notes
AUDIO_CACHE_SIZE = 4  # decoded songs kept in memory per DataLoader worker process


@dataclass
class NoteRecord:
    wav_path: Path
    start_time: float
    end_time: float
    midi_num: int
    is_bending: bool
    is_vibrt: bool
    is_breath: bool


def find_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    labels_dir = split_dir / "labels"
    audio_dir = split_dir / "audio"
    pairs = []
    for json_path in labels_dir.rglob("*.json"):
        rel = json_path.relative_to(labels_dir)
        wav_path = audio_dir / rel.with_suffix(".wav")
        if wav_path.exists():
            pairs.append((wav_path, json_path))
    return pairs


def build_note_index(pairs: list[tuple[Path, Path]]) -> list[NoteRecord]:
    records: list[NoteRecord] = []
    for wav_path, json_path in pairs:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for note in data.get("notes", []):
            start = float(note["start_time"])
            end = float(note["end_time"])
            if end - start < MIN_NOTE_DURATION_SEC:
                continue
            records.append(
                NoteRecord(
                    wav_path=wav_path,
                    start_time=start,
                    end_time=end,
                    midi_num=int(note["midi_num"]),
                    is_bending=bool(note["is_bending"]),
                    is_vibrt=bool(note["is_vibrt"]),
                    is_breath=bool(note["is_breath"]),
                )
            )
    return records


_audio_cache: "OrderedDict[Path, tuple[np.ndarray, int]]" = OrderedDict()


def _load_full_song(wav_path: Path) -> tuple[np.ndarray, int]:
    cached = _audio_cache.get(wav_path)
    if cached is not None:
        _audio_cache.move_to_end(wav_path)
        return cached

    y, native_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    _audio_cache[wav_path] = (y, native_sr)
    if len(_audio_cache) > AUDIO_CACHE_SIZE:
        _audio_cache.popitem(last=False)
    return y, native_sr


def _load_note_audio(record: NoteRecord, sr: int = TARGET_SR) -> np.ndarray:
    full_audio, native_sr = _load_full_song(record.wav_path)
    start_frame = int(record.start_time * native_sr)
    end_frame = int(record.end_time * native_sr)
    y = full_audio[start_frame:end_frame] if start_frame < len(full_audio) else np.empty(0, dtype=np.float32)

    if len(y) == 0:
        # label timestamp falls beyond the actual audio duration (rare data mismatch) -
        # return near-silence of the expected length rather than crashing the resample below
        target_len = max(1, int(round((record.end_time - record.start_time) * sr)))
        return np.zeros(target_len, dtype=np.float32)

    if native_sr != sr:
        # lightweight resample without adding a librosa dependency to ml/
        duration = len(y) / native_sr
        target_len = max(1, int(round(duration * sr)))
        x_old = np.linspace(0, 1, num=len(y), endpoint=False)
        x_new = np.linspace(0, 1, num=target_len, endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)

    return y


_mel_transform = T.MelSpectrogram(sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
_db_transform = T.AmplitudeToDB(stype="power")


def _to_log_mel(y: np.ndarray) -> torch.Tensor:
    waveform = torch.from_numpy(y).unsqueeze(0)  # (1, num_samples)
    mel = _mel_transform(waveform)
    log_mel = _db_transform(mel)  # (1, n_mels, frames)

    frames = log_mel.shape[-1]
    if frames < MAX_FRAMES:
        pad = MAX_FRAMES - frames
        log_mel = torch.nn.functional.pad(log_mel, (0, pad), mode="constant", value=float(log_mel.min()))
    else:
        log_mel = log_mel[..., :MAX_FRAMES]

    return log_mel


class SingingNoteDataset(Dataset):
    def __init__(self, split_dir: Path):
        self.split_dir = split_dir
        self.pairs = find_pairs(split_dir)
        self.index = build_note_index(self.pairs)

        groups: dict[Path, list[int]] = defaultdict(list)
        for i, record in enumerate(self.index):
            groups[record.wav_path].append(i)
        self.file_groups: list[list[int]] = list(groups.values())

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.index[idx]
        y = _load_note_audio(record)
        log_mel = _to_log_mel(y)

        midi_target = torch.tensor(float(record.midi_num), dtype=torch.float32)
        technique_target = torch.tensor(
            [float(record.is_bending), float(record.is_vibrt), float(record.is_breath)],
            dtype=torch.float32,
        )
        return log_mel, midi_target, technique_target


class FileClusteredSampler(Sampler[int]):
    """Shuffles song files, then notes within each file - keeps notes from the
    same file close together in the yielded order (unlike a plain random
    shuffle) so `_load_full_song`'s per-worker cache stays warm."""

    def __init__(self, dataset: SingingNoteDataset):
        self.file_groups = dataset.file_groups

    def __iter__(self):
        file_order = torch.randperm(len(self.file_groups)).tolist()
        for file_idx in file_order:
            group = self.file_groups[file_idx]
            for i in torch.randperm(len(group)).tolist():
                yield group[i]

    def __len__(self) -> int:
        return sum(len(g) for g in self.file_groups)
