export interface AnalysisMetrics {
  duration_sec: number;
  pitch: {
    mean_cents_deviation: number;
    pitch_stability_std_semitones: number;
    voiced_ratio: number;
    vibrato_detected_ratio: number;
    bending_detected_ratio: number;
    breath_detected_ratio: number;
  };
  rhythm: {
    estimated_tempo_bpm: number;
    onset_count: number;
    ioi_std_ms: number;
    beat_consistency_cv: number;
  };
  tone: {
    mean_hnr_db: number;
    jitter_local_pct: number;
    shimmer_local_pct: number;
  };
  dynamics: {
    dynamic_range_db: number;
    rms_std_db: number;
    phrase_loudness_trend: string;
  };
}

export interface AxisCommentary {
  pitch: string;
  rhythm: string;
  tone: string;
  dynamics: string;
  expressiveness: string;
}

export interface VocalReport {
  strengths: string[];
  improvements: string[];
  skill_level: {
    label: string;
    rationale: string;
  };
  axis_commentary: AxisCommentary;
}

export interface AnalysisResult {
  analysis_id: string;
  metrics: AnalysisMetrics;
  report: VocalReport;
}
