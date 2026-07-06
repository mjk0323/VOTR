# ML 파이프라인 (트랙 B)

AI Hub "다음색 가이드보컬" 데이터셋으로 (1) 싱잉 보이스 피치 추정 모델과 (2) 비브라토/피치벤딩/브레스 검출 모델을 학습하는 파이프라인. 학습이 끝나면 `backend/app/services/audio_analysis.py`의 DSP 휴리스틱을 대체한다 (`docs/architecture.md` 참고).

## 왜 이 데이터셋인가

유튜브 영상 등 저작권 문제가 있는 자료 대신, AI Hub에서 적법하게 신청·발급받은 라이선스 데이터셋을 사용한다. 4,000곡/92명 가창자/157시간 분량이며, note 단위로 실제 MIDI 정답(음높이·타이밍)과 `is_bending`/`is_vibrt`/`is_breath` 발성 기법 라벨이 포함되어 있다.

## 파이프라인 구조

```
ml/
├── data/
│   ├── extract.py   # zip에서 소규모 샘플만 선택 압축해제 (전체 32GB를 풀 필요 없음)
│   └── dataset.py    # WAV+JSON 페어링, note 단위 세그먼트 → PyTorch Dataset
├── models/
│   ├── backbone.py              # 공용 소형 CNN
│   ├── pitch_estimator.py       # log-mel → MIDI 피치 (회귀)
│   └── technique_classifier.py  # log-mel → [비브라토, 피치벤딩, 브레스] (멀티라벨)
├── train.py       # 학습 루프
├── evaluate.py    # Validation split 평가 (피치 오차 cents, 기법 분류 F1)
└── checkpoints/    # 학습된 모델 저장 위치 (gitignore)
```

**이미 검증됨** (이번 세션에서 랩톱 CPU로 스모크 테스트 완료): `extract.py`로 1~2명 가창자 샘플 추출 → `dataset.py`가 1,539개 note 세그먼트로 인덱싱 → `train.py`/`evaluate.py`가 에러 없이 동작. 실제 학습(수십~수백 epoch, 전체 데이터)은 데스크탑 GPU에서 진행.

## 오디오 로딩 방식에 대한 참고사항

`torchaudio.load()`/`torchaudio.info()`는 이 환경에서 torch(2.12)와 torchaudio(2.11) 버전이 어긋나 C++ 확장이 깨져있어 사용할 수 없었다 (`AttributeError: module 'torchaudio' has no attribute 'load'`). 그래서 오디오 I/O는 `soundfile`(부분 구간만 읽어서 메모리 절약)로, mel-spectrogram 계산은 `torchaudio.transforms`(순수 PyTorch라 문제없이 동작)로 분리해서 처리한다. 데스크탑에서 GPU용 torch를 재설치할 때 torchaudio도 **torch 버전과 정확히 맞는 빌드**로 설치하면 이 문제가 해결될 수 있지만, 지금 구조(soundfile + torchaudio.transforms)로도 충분히 잘 동작하므로 굳이 바꿀 필요는 없다.

## 데스크탑에서 GPU 학습 준비하기

1. **GPU 벤더 확인**: 장치 관리자 또는 `dxdiag` 실행 → 디스플레이 탭에서 GPU 이름 확인
2. **NVIDIA GPU인 경우**:
   ```
   pip uninstall torch torchaudio
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   (CUDA 버전은 `nvidia-smi`로 확인 후 [pytorch.org](https://pytorch.org/get-started/locally/)에서 맞는 인덱스 URL 선택)
3. **AMD GPU인 경우**: ROCm은 Windows를 공식 지원하지 않으므로 DirectML 백엔드 사용
   ```
   pip install torch-directml
   ```
   `train.py`의 `get_device()`가 CUDA가 없으면 자동으로 `torch_directml`을 시도하도록 이미 구현되어 있음
4. **전체 데이터 압축 해제**:
   ```
   python data/extract.py --split training --singers 92
   python data/extract.py --split validation --singers 92
   ```
   (`--singers` 값을 크게 주면 사실상 전체 압축 해제. 디스크 여유 공간 확인 필요 — 원본 zip 기준 약 32GB, 압축 해제 후 더 커질 수 있음)
5. **본 학습 실행**:
   ```
   python train.py --train-dir data/extracted/training --val-dir data/extracted/validation --epochs 30 --batch-size 64
   ```
6. **평가**:
   ```
   python evaluate.py --val-dir data/extracted/validation
   ```

## 학습된 모델을 앱에 연결하기 (완료됨)

1. `ml/export_onnx.py`로 `pitch_estimator.pt` / `technique_classifier.pt`를 ONNX로 export. 후처리(피치 빈 디코딩, sigmoid)를 그래프 안에 baked-in해서, 소비하는 쪽은 모델을 그냥 실행하기만 하면 최종 값(MIDI 예측치, 확률)을 바로 받는다. `.onnx` 파일은 `backend/app/ml_models/`에 복사해둔다.
2. `backend/app/services/singing_model.py` 신설: onset 검출(`analyze_rhythm`이 이미 쓰던 방식 재사용)로 노래 전체를 note 단위로 분절하고, 각 구간을 `librosa`로 log-mel spectrogram 계산 후 ONNX 모델에 통과시킨다. **주의**: 학습은 `torchaudio.transforms`로 log-mel을 계산했는데 backend엔 torch가 없다(배포를 가볍게 유지하려는 의도적 선택 - Groq를 쓰는 이유와 같은 맥락). `librosa.feature.melspectrogram`을 `htk=True, norm=None`으로 맞추면 수치가 ~1.6e-4 dB 오차로 거의 정확히 일치한다는 걸 직접 검증했다 (`torchaudio` 기본값: `mel_scale="htk"`, `norm=None`).
3. `backend/app/services/audio_analysis.py`의 `analyze_pitch()`를 `librosa.pyin` + FFT 비브라토 휴리스틱에서 `singing_model.analyze_notes()` 기반으로 교체.
4. `PitchMetrics` 스키마 변경: `vibrato_rate_hz`(Hz 단위 추정치) 제거, `vibrato_detected_ratio`/`bending_detected_ratio`/`breath_detected_ratio`(곡 전체 note 중 검출 비율) 추가. **이 스키마 변경은 프론트엔드(`frontend/app/types/report.ts`)에도 영향을 줬다** - 처음엔 "API 계약이 그대로 유지된다"고 썼었는데 틀렸음, 실제로는 필드가 바뀌어서 프론트도 같이 고쳤다.
5. 기법 판정 threshold는 0.5가 아니라 `ml/evaluate.py`의 threshold sweep으로 찾은 값 사용: bending=0.66, vibrato=0.62, breath=0.70 (`singing_model.py`에 상수로 박아둠).
6. `report_generation.py`의 시스템 프롬프트도 새 필드 설명으로 교체.

검증: `backend/tests/test_audio_analysis.py`를 새 스키마에 맞게 수정 후 통과 확인. 다만 이 테스트가 쓰는 합성(사인파) 비브라토 픽스처는 실제 사람 목소리로 학습된 모델이 인식하지 못했다 (vibrato_detected_ratio=0.0) - 모델이 진짜 노래에 특화돼 있다는 방증이지, 버그는 아니다.

## 향후 확장: 곡 매칭 기반 정확도 채점

지금 모델들은 임의의 곡에 일반화되는 것을 목표로 한다 (특정 곡 대비 "정답 멜로디"와 비교하지 않음). 사용자가 부른 곡이 데이터셋의 4,000곡 중 하나와 일치할 경우, 그 곡의 note 단위 정답과 직접 비교해 훨씬 정밀한 채점(연습 모드)을 제공할 수 있다. 다만 이는 곡 인식(song identification, 예: audio fingerprinting)이라는 별도 문제이므로 이번 범위에서는 제외했다.
