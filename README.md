# VOTR — AI 보컬 평가

노래를 녹음하거나 업로드하면 AI가 음정, 박자, 톤, 다이나믹스, 표현력을 분석해 잘한 점/고칠 점/현재 수준을 담은 리포트를 만들어주는 앱.

## 구성

- **트랙 A (`frontend/`, `backend/`)**: 실제 동작하는 앱. Next.js 프론트엔드 + FastAPI 백엔드. 오디오를 신호처리(librosa/parselmouth)로 분석하고, 그 수치를 Groq LLM에 전달해 리포트를 생성한다.
- **트랙 B (`ml/`)**: AI Hub "다음색 가이드보컬" 데이터셋으로 싱잉 전용 피치 추정/발성 기법(비브라토·피치벤딩·브레스) 검출 모델을 학습하는 파이프라인. 완성되면 트랙 A의 DSP 휴리스틱을 대체한다.

자세한 설계는 [docs/architecture.md](docs/architecture.md), 평가 루브릭은 [docs/rubric.md](docs/rubric.md), ML 파이프라인은 [docs/ml_pipeline.md](docs/ml_pipeline.md) 참고.

## 빠른 시작

**프론트엔드**
```
cd frontend
npm install
npm run dev
```

**백엔드**
```
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # GROQ_API_KEY 입력 (무료, console.groq.com에서 발급)
uvicorn app.main:app --reload
```

기본적으로 프론트엔드는 `http://localhost:3000`, 백엔드는 `http://localhost:8000`에서 동작한다.

## 기술 스택

- 프론트엔드: Next.js (App Router) + TypeScript + Tailwind CSS
- 백엔드: FastAPI + librosa + praat-parselmouth + imageio-ffmpeg
- 리포트 생성: Groq (무료 클라우드 LLM 추론)
- ML(트랙 B): PyTorch + torchaudio
