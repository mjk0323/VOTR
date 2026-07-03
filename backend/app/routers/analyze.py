from fastapi import APIRouter, UploadFile
from pydantic import BaseModel

from app.schemas.metrics import AnalysisMetrics
from app.schemas.report import VocalReport
from app.services import media_ingest
from app.services.audio_analysis import analyze_audio
from app.services.report_generation import generate_report

router = APIRouter()


class AnalysisResult(BaseModel):
    analysis_id: str
    metrics: AnalysisMetrics
    report: VocalReport


@router.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile):
    media_type = media_ingest.classify_media(file)
    raw_bytes = await media_ingest.read_upload_with_limit(file, media_type)

    analysis_id, analysis_dir = media_ingest.new_analysis_dir()
    wav_path = media_ingest.extract_normalized_wav(raw_bytes, file.filename or "upload", analysis_dir)

    metrics = analyze_audio(wav_path)
    report = generate_report(metrics)

    (analysis_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2), encoding="utf-8")
    (analysis_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

    return AnalysisResult(analysis_id=analysis_id, metrics=metrics, report=report)
