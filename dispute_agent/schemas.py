from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DisputeType(str, Enum):
    NOT_RECEIVED = "not_received"
    DUPLICATE_CHARGE = "duplicate_charge"
    UNAUTHORIZED = "unauthorized"
    NOT_AS_DESCRIBED = "not_as_described"
    SUBSCRIPTION_CANCELLED = "subscription_cancelled"


class Action(str, Enum):
    REFUND_FULL = "refund_full"
    DENY = "deny"
    REQUEST_EVIDENCE = "request_evidence"
    ESCALATE_HUMAN = "escalate_human"


class Label(BaseModel):

    dispute_type: DisputeType
    action: Action
    refund_amount: float = 0.0
    policy_ids: list[str]


class Case(BaseModel):

    case_id: str
    customer_id: str
    txn_id: str
    created_at: str
    complaint: str
    merchant_response: str | None = None
    attachments: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class LabeledCase(BaseModel):
    case: Case
    label: Label


class Decision(BaseModel):

    dispute_type: DisputeType
    action: Action
    refund_amount: float = Field(0.0, ge=0)
    policy_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = Field(0.5, ge=0, le=1)


class GuardedDecision(BaseModel):
    raw: Decision
    final: Decision
    guardrails_triggered: list[str] = Field(default_factory=list)


class RunTrace(BaseModel):

    case_id: str
    decision: GuardedDecision | None
    tool_calls: list[dict] = Field(default_factory=list)
    seen_evidence_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
