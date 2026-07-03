"""DSP-based acoustic feature extraction for vocal evaluation.

Computes objective metrics per axis (pitch, rhythm, tone, dynamics) from a
normalized mono WAV. These feed report_generation.py, which turns them into
qualitative feedback. Track B (ml/) will later replace analyze_pitch's F0
estimation and add technique detection (vibrato/bending/breath) trained on
real singing data instead of the heuristics below.
"""

from pathlib import Path

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.ndimage import median_filter

from app.schemas.metrics import AnalysisMetrics, DynamicsMetrics, PitchMetrics, RhythmMetrics, ToneMetrics

PITCH_FMIN = librosa.note_to_hz("C2")
PITCH_FMAX = librosa.note_to_hz("C7")
VIBRATO_BAND_HZ = (4.0, 8.0)


def _estimate_vibrato_rate(detrended_semitones: np.ndarray, frame_rate: float) -> float | None:
    n = len(detrended_semitones)
    if n < frame_rate:  # need at least ~1s of voiced signal to resolve 4-8Hz band
        return None

    windowed = detrended_semitones * np.hanning(n)
    fft_mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / frame_rate)

    band_mask = (freqs >= VIBRATO_BAND_HZ[0]) & (freqs <= VIBRATO_BAND_HZ[1])
    if not np.any(band_mask):
        return None

    band_freqs = freqs[band_mask]
    band_mag = fft_mag[band_mask]
    peak_idx = int(np.argmax(band_mag))

    overall_mean = float(np.mean(fft_mag)) + 1e-9
    if band_mag[peak_idx] < 3 * overall_mean:
        return None

    return round(float(band_freqs[peak_idx]), 2)


def analyze_pitch(y: np.ndarray, sr: int) -> PitchMetrics:
    f0, _voiced_flag, _voiced_prob = librosa.pyin(y, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=sr)
    voiced_mask = ~np.isnan(f0)
    voiced_ratio = float(np.mean(voiced_mask)) if len(f0) else 0.0

    if not np.any(voiced_mask):
        return PitchMetrics(
            mean_cents_deviation=0.0,
            pitch_stability_std_semitones=0.0,
            voiced_ratio=voiced_ratio,
            vibrato_rate_hz=None,
        )

    midi_voiced = librosa.hz_to_midi(f0[voiced_mask])

    nearest_semitone = np.round(midi_voiced)
    cents_deviation = (midi_voiced - nearest_semitone) * 100
    mean_cents_deviation = float(np.mean(np.abs(cents_deviation)))

    hop_length = 512
    frame_rate = sr / hop_length
    window = max(3, int(round(0.5 * frame_rate)))
    if window % 2 == 0:
        window += 1
    local_median = median_filter(midi_voiced, size=window, mode="nearest")
    detrended = midi_voiced - local_median
    pitch_stability_std = float(np.std(detrended))

    return PitchMetrics(
        mean_cents_deviation=round(mean_cents_deviation, 1),
        pitch_stability_std_semitones=round(pitch_stability_std, 3),
        voiced_ratio=round(voiced_ratio, 3),
        vibrato_rate_hz=_estimate_vibrato_rate(detrended, frame_rate),
    )


def analyze_rhythm(y: np.ndarray, sr: int) -> RhythmMetrics:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    if len(onset_times) < 2:
        return RhythmMetrics(
            estimated_tempo_bpm=round(tempo_bpm, 1),
            onset_count=int(len(onset_times)),
            ioi_std_ms=0.0,
            beat_consistency_cv=0.0,
        )

    iois_ms = np.diff(onset_times) * 1000
    ioi_std_ms = float(np.std(iois_ms))
    ioi_mean_ms = float(np.mean(iois_ms))
    beat_consistency_cv = ioi_std_ms / ioi_mean_ms if ioi_mean_ms > 0 else 0.0

    return RhythmMetrics(
        estimated_tempo_bpm=round(tempo_bpm, 1),
        onset_count=int(len(onset_times)),
        ioi_std_ms=round(ioi_std_ms, 1),
        beat_consistency_cv=round(beat_consistency_cv, 3),
    )


def analyze_tone(wav_path: Path) -> ToneMetrics:
    sound = parselmouth.Sound(str(wav_path))

    harmonicity = sound.to_harmonicity()
    hnr_values = harmonicity.values.flatten()
    hnr_values = hnr_values[hnr_values != -200]  # -200 = Praat's "undefined" sentinel
    mean_hnr_db = float(np.mean(hnr_values)) if len(hnr_values) else 0.0

    jitter_local = 0.0
    shimmer_local = 0.0
    try:
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = call(
            [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
    except Exception:
        pass

    if jitter_local is None or np.isnan(jitter_local):
        jitter_local = 0.0
    if shimmer_local is None or np.isnan(shimmer_local):
        shimmer_local = 0.0

    return ToneMetrics(
        mean_hnr_db=round(mean_hnr_db, 2),
        jitter_local_pct=round(jitter_local * 100, 3),
        shimmer_local_pct=round(shimmer_local * 100, 3),
    )


def analyze_dynamics(y: np.ndarray, sr: int) -> DynamicsMetrics:
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    valid = rms_db[np.isfinite(rms_db)]

    if len(valid) == 0:
        return DynamicsMetrics(dynamic_range_db=0.0, rms_std_db=0.0, phrase_loudness_trend="stable")

    dynamic_range_db = float(np.max(valid) - np.min(valid))
    rms_std_db = float(np.std(valid))

    n = len(valid)
    first_half_mean = float(np.mean(valid[: max(1, n // 2)]))
    second_half_mean = float(np.mean(valid[n // 2 :])) if n >= 2 else first_half_mean
    diff = second_half_mean - first_half_mean

    if diff > 1.5:
        trend = "increasing"
    elif diff < -1.5:
        trend = "decreasing"
    else:
        trend = "stable"

    return DynamicsMetrics(
        dynamic_range_db=round(dynamic_range_db, 1),
        rms_std_db=round(rms_std_db, 2),
        phrase_loudness_trend=trend,
    )


def analyze_audio(wav_path: Path) -> AnalysisMetrics:
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    duration_sec = round(float(len(y) / sr), 2)

    return AnalysisMetrics(
        duration_sec=duration_sec,
        pitch=analyze_pitch(y, sr),
        rhythm=analyze_rhythm(y, sr),
        tone=analyze_tone(wav_path),
        dynamics=analyze_dynamics(y, sr),
    )
