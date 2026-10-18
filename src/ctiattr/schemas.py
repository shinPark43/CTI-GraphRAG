"""Pydantic v2 data models: graph entities, report-layer records, evidence cards,
gold triples and the attribution API payloads (design document, Section 8).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from pydantic_core import PydanticCustomError

ID_PATTERN = r"^(T\d{4}(\.\d{3})?|CWE-\d+|CAPEC-\d+|CVE-\d{4}-\d{4,})$"
TECHNIQUE_PATTERN = r"^T\d{4}(\.\d{3})?$"


# --- graph ----------------------------------------------------------------

class Taxonomy(str, Enum):
    ATTACK = "attack"   # ATA: MITRE ATT&CK techniques
    CWE = "cwe"         # VCA: CWE weaknesses


class NodeLabel(str, Enum):
    TECHNIQUE = "AttackTechnique"
    CWE = "CWE"
    CAPEC = "CAPEC"
    CVE = "CVE"


class KBNode(BaseModel):
    id: str = Field(..., pattern=ID_PATTERN, description="Snapshot identifier, e.g. T1059.001 or CWE-79")
    label: NodeLabel
    title: str
    text: str = Field("", description="Snapshot `contents`, CAPEC description or CVE description")
    domain: str | None = Field(None, description="ATT&CK only (rule R1): enterprise | mobile | ics")
    replaced_by: str | None = Field(None, description="ATT&CK only (rule R2): current ID of a retired technique")
    tactics: list[str] = Field(default_factory=list, description="ATT&CK only, from kill_chain_phases")


class Predicate(str, Enum):
    SUBTECHNIQUE_OF = "SUBTECHNIQUE_OF"
    IN_TACTIC = "IN_TACTIC"
    REPLACED_BY = "REPLACED_BY"
    EXPLOITS_WEAKNESS = "EXPLOITS_WEAKNESS"
    MAPS_TO_TECHNIQUE = "MAPS_TO_TECHNIQUE"
    HAS_WEAKNESS = "HAS_WEAKNESS"
    GROUNDS = "GROUNDS"
    DESCRIBES_TECHNIQUE = "DESCRIBES_TECHNIQUE"
    DESCRIBES_WEAKNESS = "DESCRIBES_WEAKNESS"


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    predicate: Predicate
    origin: str = Field(..., description='"corpus_kb", "report_layer" or "rule:R<n>"')
    method: Literal["asserted", "derived", "segmented", "extracted"] = "asserted"
    confidence: float | None = Field(None, ge=0, le=1, description="1.0 unless extracted; then the audit "
                                                                    "precision of the edge's tier")
    tier: Literal["supported", "context", "conflict"] | None = None


# --- report layer -----------------------------------------------------------

class ReportChunk(BaseModel):
    chunk_id: str = Field(..., pattern=r"^BLOG-\d+#\d+$")
    report_id: str = Field(..., pattern=r"^BLOG-\d+$")
    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)
    text: str


class ExtractedTTP(BaseModel):
    """One record of the extraction reply, exactly as the LLM must produce it."""
    model_config = ConfigDict(extra="forbid")
    behavior: str = Field(..., min_length=1, description="The verbatim attacker-behavior phrase from the chunk")
    technique_id: str = Field(..., pattern=TECHNIQUE_PATTERN, description="ATT&CK ID, e.g. T1486 or T1059.001")
    technique_name: str = Field(..., description="Official ATT&CK technique name")

    @field_validator("behavior", mode="before")
    @classmethod
    def _squash(cls, v):
        return " ".join(v.split()) if isinstance(v, str) else v

    @field_validator("technique_id", mode="before")
    @classmethod
    def _upper(cls, v):
        return v.strip().upper() if isinstance(v, str) else v


class ExtractionReply(BaseModel):
    """The whole reply for one chunk; its JSON schema constrains generation (report_layer.reply_schema)."""
    model_config = ConfigDict(extra="forbid")
    ttps: list[ExtractedTTP]


class GroundedBehavior(BaseModel):
    """One report-layer record: a verbatim behavior phrase grounded to a technique.

    Validated with context {"chunk_text": str, "technique_ids": set[str]}, it also
    enforces grounding: the ID exists in the snapshot (error type `unknown_id`) and the
    phrase occurs verbatim, whitespace-normalized, in the chunk (`not_verbatim`).
    """
    chunk_id: str
    report_id: str
    behavior: str = Field(..., min_length=1, description="Verbatim substring of the chunk")
    technique_id: str = Field(..., pattern=TECHNIQUE_PATTERN)
    technique_name: str = ""

    @model_validator(mode="after")
    def _grounded(self, info: ValidationInfo):
        ctx = info.context or {}
        if "technique_ids" in ctx and self.technique_id not in ctx["technique_ids"]:
            raise PydanticCustomError("unknown_id", "{tid} is not in the snapshot", {"tid": self.technique_id})
        if "chunk_text" in ctx and self.behavior not in " ".join(ctx["chunk_text"].split()):
            raise PydanticCustomError("not_verbatim", "behavior is not a verbatim span of the chunk")
        return self


# --- evaluation -------------------------------------------------------------

class GoldTriple(BaseModel):
    id: str
    subject_id: str = Field(..., pattern=r"^BLOG-\d+$")
    predicate: Predicate
    object_id: str = Field(..., pattern=ID_PATTERN)
    evidence_text: str
    source_question: str


# --- engine and API ---------------------------------------------------------

class Candidate(BaseModel):
    id: str = Field(..., pattern=ID_PATTERN)
    rank: int = Field(..., ge=1)
    card: str = Field(..., description="Evidence card shown to the model")
    sources: list[str] = Field(default_factory=list, description='e.g. "dense_question", "capec", "cve", "report"')


class LLMCall(BaseModel):
    step: str = Field(..., description='"select" or "explain"')
    prompt: str
    output: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float


class AttributionRequest(BaseModel):
    passage: str = Field(..., min_length=10)
    taxonomy: Taxonomy
    question: str | None = Field(None, description="Full user question; defaults to the passage")
    report_id: str | None = Field(None, description="Source report, excluded from report-layer examples")
    method: str = Field("sgs", description='"cb" | "vr" | "sgs"')
    explain: bool = True


class AttributionResponse(BaseModel):
    answer_id: str = Field(..., pattern=ID_PATTERN, description="Exactly one identifier")
    answer_title: str
    explanation: str = Field("", description="Contains no identifier other than answer_id")
    candidates: list[Candidate] = Field(default_factory=list)
    votes: list[str] = Field(default_factory=list, description="VCA: the pick of each of the 5 rotations")
    calls: list[LLMCall] = Field(default_factory=list)
    model: str
    latency_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
