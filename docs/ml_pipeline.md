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

## 학습된 모델을 앱에 연결하기 (완료 후)

1. `pitch_estimator.pt` / `technique_classifier.pt`를 ONNX로 export (`torch.onnx.export`)
2. `backend/app/services/audio_analysis.py`의 `analyze_pitch()`가 `librosa.pyin` 대신 ONNX 모델 추론을 사용하도록 교체 (`onnxruntime`을 backend 쪽 venv에 추가 설치)
3. `technique_classifier`의 출력(비브라토/벤딩/브레스 확률)을 `PitchMetrics`/새 필드로 추가해 `report_generation.py` 프롬프트에 반영

이 교체는 `audio_analysis.py` 내부 구현만 바뀌는 것이라, 프론트엔드/API 계약/`report_generation.py`는 그대로 유지된다.

## 향후 확장: 곡 매칭 기반 정확도 채점

지금 모델들은 임의의 곡에 일반화되는 것을 목표로 한다 (특정 곡 대비 "정답 멜로디"와 비교하지 않음). 사용자가 부른 곡이 데이터셋의 4,000곡 중 하나와 일치할 경우, 그 곡의 note 단위 정답과 직접 비교해 훨씬 정밀한 채점(연습 모드)을 제공할 수 있다. 다만 이는 곡 인식(song identification, 예: audio fingerprinting)이라는 별도 문제이므로 이번 범위에서는 제외했다.
