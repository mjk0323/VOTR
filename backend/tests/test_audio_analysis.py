from pathlib import Path

from app.services.audio_analysis import analyze_audio

FIXTURE = Path(__file__).parent / "fixtures" / "sample_clip.wav"


def test_analyze_audio_returns_plausible_metrics():
    metrics = analyze_audio(FIXTURE)

    assert 4.5 < metrics.duration_sec < 5.5

    assert 0.0 <= metrics.pitch.voiced_ratio <= 1.0
    assert metrics.pitch.mean_cents_deviation >= 0.0
    assert metrics.pitch.pitch_stability_std_semitones >= 0.0

    # synthetic fixture has a clear ~5.5Hz vibrato within the detectable 4-8Hz band
    assert metrics.pitch.vibrato_rate_hz is not None
    assert 4.0 <= metrics.pitch.vibrato_rate_hz <= 8.0

    assert 40 <= metrics.rhythm.estimated_tempo_bpm <= 240
    assert metrics.rhythm.onset_count >= 0

    assert metrics.tone.mean_hnr_db > 0
    assert metrics.tone.jitter_local_pct >= 0.0
    assert metrics.tone.shimmer_local_pct >= 0.0

    assert metrics.dynamics.dynamic_range_db >= 0.0
    assert metrics.dynamics.phrase_loudness_trend in {"increasing", "decreasing", "stable"}
