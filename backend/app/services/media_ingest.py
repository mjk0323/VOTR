import subprocess
import uuid
from pathlib import Path
from typing import Literal

import imageio_ffmpeg
from fastapi import HTTPException, UploadFile

from app.config import STORAGE_DIR, settings

MediaType = Literal["audio", "video"]

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def classify_media(upload: UploadFile) -> MediaType:
    content_type = upload.content_type or ""
    if content_type.startswith("video/"):
        return "video"
    if content_type.startswith("audio/"):
        return "audio"

    suffix = Path(upload.filename or "").suffix.lower()
    if suffix in _VIDEO_EXTENSIONS:
        return "video"
    return "audio"


async def read_upload_with_limit(upload: UploadFile, media_type: MediaType) -> bytes:
    max_mb = settings.max_video_upload_mb if media_type == "video" else settings.max_audio_upload_mb
    max_bytes = max_mb * 1024 * 1024

    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{'영상' if media_type == 'video' else '오디오'} 파일은 최대 {max_mb}MB까지 업로드할 수 있습니다.",
            )
        chunks.append(chunk)

    if total == 0:
        raise HTTPException(status_code=400, detail="빈 파일은 업로드할 수 없습니다.")

    return b"".join(chunks)


def extract_normalized_wav(raw_bytes: bytes, upload_filename: str, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    source_suffix = Path(upload_filename).suffix or ".bin"
    source_path = work_dir / f"source{source_suffix}"
    source_path.write_bytes(raw_bytes)

    output_path = work_dir / "normalized.wav"
    cmd = [
        _FFMPEG_EXE,
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(settings.target_sample_rate),
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        raise HTTPException(
            status_code=422,
            detail="오디오를 처리할 수 없습니다. 파일 형식을 확인해주세요.",
        )
    return output_path


def new_analysis_dir() -> tuple[str, Path]:
    analysis_id = uuid.uuid4().hex
    analysis_dir = STORAGE_DIR / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_id, analysis_dir
