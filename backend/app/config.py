from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BACKEND_DIR / "storage"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    cors_origins: list[str] = ["http://localhost:3000"]

    max_audio_upload_mb: int = 20
    max_video_upload_mb: int = 100

    target_sample_rate: int = 22050


settings = Settings()
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
