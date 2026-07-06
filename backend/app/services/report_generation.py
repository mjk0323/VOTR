"""Turns computed DSP metrics into a qualitative vocal coaching report via Groq
(free cloud inference of open-weight models - fast enough and scalable enough
for a multi-user public deployment, unlike CPU-only local Ollama).

The model never sees raw audio - only the structured metrics from
audio_analysis.py. This keeps objective measurements (pitch/rhythm/tone/
dynamics) grounded in real signal processing while letting the LLM synthesize
coach-style feedback from them. Groq's JSON mode guarantees valid JSON but not
a specific schema, so the exact schema is spelled out in the system prompt and
the response is validated against VocalReport (with one retry on failure).
"""

import json
import re

import groq
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import settings
from app.schemas.metrics import AnalysisMetrics
from app.schemas.report import VocalReport

REPORT_JSON_SCHEMA = VocalReport.model_json_schema()

SYSTEM_PROMPT = f"""\
당신은 다양한 장르를 가르치는 숙련된 보컬 코치입니다. 사용자가 부른 노래를 신호처리로 \
분석한 "객관적 수치"가 주어집니다. 이 수치는 실제로 계산된 값이며 추측이 아닙니다. \
당신의 역할은 이 수치들을 보컬 교육학적으로 해석해 구체적이고 건설적인 피드백을 작성하는 것입니다.

평가 축과 해석 기준:
- 음정(pitch): mean_cents_deviation(가장 가까운 반음 대비 이탈, cents)이 클수록 음정이 불안정. \
pitch_stability_std_semitones가 클수록 곡 전체에서 음정이 들쭉날쭉함(음마다 어긋나는 방향/정도가 제각각). \
vibrato_detected_ratio가 높을수록 비브라토를 자주 사용. bending_detected_ratio가 높을수록 음을 미끄러지듯 \
굴리는 피치 벤딩을 자주 사용. breath_detected_ratio가 높을수록 숨소리가 섞인 발성(브레시니스)을 자주 사용.
- 박자(rhythm): 참조 트랙(정답 멜로디)이 없으므로 "정확도"가 아니라 "자기 안정성"만 측정된 값입니다. \
beat_consistency_cv가 낮을수록 박자가 일정함. 이 축에 대해서는 과도하게 단정적으로 "박자가 틀렸다"고 \
말하지 말고 "안정성" 관점으로만 서술하세요.
- 톤(tone): mean_hnr_db가 높을수록 소리가 깨끗함(숨소리/잡음이 적음). jitter/shimmer가 낮을수록 발성이 안정적.
- 다이나믹스(dynamics): dynamic_range_db가 클수록 강약 표현의 폭이 넓음. phrase_loudness_trend로 \
곡 전개에 따른 볼륨 변화 경향을 알 수 있음.
- 감정/표현(expressiveness): 별도의 감정인식 수치는 없습니다. 위 지표들(다이나믹스 변화, 비브라토 사용, \
피치 안정성)을 종합해 표현이 단조로운지 다채로운지 해석적으로만 서술하세요. 이 축은 다른 축보다 훨씬 \
더 해석적 판단임을 명심하고, 과도하게 단정적인 표현은 피하세요.

일반 원칙:
- 주어진 수치가 뒷받침하지 않는 확신에 찬 주장을 하지 마세요 (예: 정확한 정답 멜로디와 비교한 것처럼 말하지 말 것).
- 항상 한국어로, 격려하되 솔직하게 작성하세요.
- 모든 텍스트 필드는 자연스러운 한국어 문장으로 작성하세요.
- mean_cents_deviation, beat_consistency_cv 같은 원본 변수명이나 영어 필드 이름을 절대 그대로 언급하지 마세요. \
"음정이 목표음보다 살짝 낮게 나는 경향이 있어요"처럼 사람이 쓰는 자연스러운 표현으로만 풀어서 설명하세요.
- 각 항목은 1~2문장으로 간결하게 작성하세요.
- 한자나 다른 외국어 문자를 절대 섞지 말고, 순수한 한글 문장으로만 작성하세요.

반드시 다음 JSON 스키마를 정확히 따르는 JSON 객체 하나로만 응답하세요 (다른 텍스트 없이):
{json.dumps(REPORT_JSON_SCHEMA, ensure_ascii=False)}
"""


def _client() -> groq.Groq:
    return groq.Groq(api_key=settings.groq_api_key)


def _call_groq(metrics: AnalysisMetrics) -> VocalReport:
    client = _client()
    user_message = (
        "다음은 사용자가 부른 노래를 신호처리로 분석한 수치입니다. 이 수치를 근거로 리포트를 JSON으로 작성해주세요.\n\n"
        f"{metrics.model_dump_json(indent=2)}"
    )

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.5,
        max_tokens=1200,
    )

    content = response.choices[0].message.content
    return VocalReport.model_validate_json(content)


_HANJA_PATTERN = re.compile(r"[一-鿿]")


def _has_hanja(report: VocalReport) -> bool:
    return bool(_HANJA_PATTERN.search(report.model_dump_json()))


def generate_report(metrics: AnalysisMetrics) -> VocalReport:
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=502,
            detail="GROQ_API_KEY가 설정되지 않았습니다. backend/.env에 무료 Groq API 키를 입력해주세요.",
        )

    try:
        report = _call_groq(metrics)
        # the free model occasionally mixes in Hanja for low-frequency Korean
        # words - one retry usually produces a clean pure-Hangul response
        if _has_hanja(report):
            report = _call_groq(metrics)
        return report
    except ValidationError:
        # malformed JSON on first attempt - retry once
        try:
            return _call_groq(metrics)
        except ValidationError as exc:
            raise HTTPException(
                status_code=502, detail="AI 리포트 생성에 실패했습니다. 잠시 후 다시 시도해주세요."
            ) from exc
    except groq.APIError as exc:
        raise HTTPException(status_code=502, detail=f"Groq API 호출에 실패했습니다: {exc.message}") from exc
