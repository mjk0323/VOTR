from pydantic import BaseModel


class SkillLevel(BaseModel):
    label: str
    rationale: str


class AxisCommentary(BaseModel):
    pitch: str
    rhythm: str
    tone: str
    dynamics: str
    expressiveness: str


class VocalReport(BaseModel):
    strengths: list[str]
    improvements: list[str]
    skill_level: SkillLevel
    axis_commentary: AxisCommentary
    # Future extension point (not built yet, see docs/architecture.md):
    # recommendations: Recommendations | None = None
