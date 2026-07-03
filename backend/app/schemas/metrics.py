from pydantic import BaseModel


class PitchMetrics(BaseModel):
    mean_cents_deviation: float
    pitch_stability_std_semitones: float
    voiced_ratio: float
    vibrato_rate_hz: float | None = None


class RhythmMetrics(BaseModel):
    estimated_tempo_bpm: float
    onset_count: int
    ioi_std_ms: float
    beat_consistency_cv: float


class ToneMetrics(BaseModel):
    mean_hnr_db: float
    jitter_local_pct: float
    shimmer_local_pct: float


class DynamicsMetrics(BaseModel):
    dynamic_range_db: float
    rms_std_db: float
    phrase_loudness_trend: str


class AnalysisMetrics(BaseModel):
    duration_sec: float
    pitch: PitchMetrics
    rhythm: RhythmMetrics
    tone: ToneMetrics
    dynamics: DynamicsMetrics
