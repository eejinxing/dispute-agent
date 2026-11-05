from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from . import guardrails, policy
from .backend import TOOL_SPECS, Backend, ToolSession
from .schemas import Action, Case, Decision, DisputeType, RunTrace

PROMPT_DIR = Path(__file__).parent / "prompts"
DEFAULT_MODEL = os.environ.get("DISPUTE_AGENT_MODEL", "claude-sonnet-5-5")
MAX_TURNS = 12

SUBMIT_SPEC: dict[str, Any] = {
    "name": "submit_decision",
    "description": "Submit the final decision for this dispute. Call exactly once, at the end.",
    "input_schema": {
        "type": "object",
        "properties": {
            "dispute_type": {"type": "string", "enum": [t.value for t in DisputeType]},
            "action": {"type": "string", "enum": [a.value for a in Action]},
            "refund_amount": {"type": "number"},
            "policy_ids": {"type": "array", "items": {"type": "string"}},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["dispute_type", "action", "refund_amount", "policy_ids", "evidence_ids", "rationale", "confidence"],
    },
}


class Agent(Protocol):
    name: str

    def run(self, case: Case, db: Backend) -> RunTrace: ...


def load_prompt(version: str) -> str:
    text = (PROMPT_DIR / f"{version}.md").read_text(encoding="utf-8")
    return text.replace("{policy}", policy.policy_text())


def render_ticket(case: Case) -> str:
    return (
        f"case_id: {case.case_id}\n"
        f"customer_id: {case.customer_id}\n"
        f"txn_id: {case.txn_id}\n"
        f"created_at (dispute date): {case.created_at}\n"
        f"attachments: {case.attachments or 'none'}\n\n"
        f"<customer_complaint untrusted=\"true\">\n{case.complaint}\n</customer_complaint>\n\n"
        f"<merchant_response untrusted=\"true\">\n{case.merchant_response or '(none)'}\n</merchant_response>"
    )


class LLMAgent:
    def __init__(self, client: Any = None, model: str = DEFAULT_MODEL, prompt_version: str = "v2"):
        if model.startswith("gemini:"):
            from .gemini_client import GeminiClient
            model = model.removeprefix("gemini:")
            client = client or GeminiClient()
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.prompt_version = prompt_version
        self.system = load_prompt(prompt_version)
        self.name = f"llm-{prompt_version}"

    def run(self, case: Case, db: Backend) -> RunTrace:
        session = ToolSession(db, case.attachments)
        messages: list[dict] = [{"role": "user", "content": render_ticket(case)}]
        trace = RunTrace(case_id=case.case_id, decision=None)
        start = time.perf_counter()
        try:
            decision = self._loop(messages, session, trace)
            if decision is None:
                trace.error = "no decision within turn limit"
                decision = Decision(dispute_type=DisputeType.NOT_RECEIVED, action=Action.ESCALATE_HUMAN,
                                    rationale="agent did not finish", confidence=0.0)
            trace.decision = guardrails.apply(decision, case, db, session.seen_evidence)
        except Exception as e:
            trace.error = f"{type(e).__name__}: {e}"
        trace.latency_s = round(time.perf_counter() - start, 2)
        trace.tool_calls = session.calls
        trace.seen_evidence_ids = sorted(session.seen_evidence)
        return trace

    def _loop(self, messages: list[dict], session: ToolSession, trace: RunTrace) -> Decision | None:
        for _ in range(MAX_TURNS):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=self.system,
                tools=TOOL_SPECS + [SUBMIT_SPEC],
                tool_choice={"type": "any"},
                messages=messages,
            )
            usage = getattr(resp, "usage", None)
            if usage is not None:
                trace.input_tokens += getattr(usage, "input_tokens", 0) or 0
                trace.output_tokens += getattr(usage, "output_tokens", 0) or 0

            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                if block.name == "submit_decision":
                    try:
                        return Decision.model_validate(block.input)
                    except ValidationError as e:
                        results.append(_tool_result(block.id, {"error": f"invalid decision: {e.errors()[:3]}"}, True))
                        continue
                out = session.call(block.name, dict(block.input))
                results.append(_tool_result(block.id, out, "error" in out))
            if not results:
                return None
            messages.append({"role": "user", "content": results})
        return None


def _tool_result(tool_use_id: str, payload: dict, is_error: bool) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(payload, ensure_ascii=False),
        "is_error": is_error,
    }


KEYWORDS: list[tuple[DisputeType, list[str]]] = [
    (DisputeType.UNAUTHORIZED, ["never made", "don't recognise", "hacked"]),
    (DisputeType.DUPLICATE_CHARGE, ["twice"]),
    (DisputeType.SUBSCRIPTION_CANCELLED, ["cancel", "subscription"]),
    (DisputeType.NOT_AS_DESCRIBED, ["broken", "listing"]),
    (DisputeType.NOT_RECEIVED, ["never arrived", "received", "parcel"]),
]


def classify_by_keywords(text: str) -> DisputeType | None:
    low = text.lower()
    for t, words in KEYWORDS:
        if any(w.lower() in low for w in words):
            return t
    return None


class NaiveBaseline:

    name = "naive-baseline"

    def run(self, case: Case, db: Backend) -> RunTrace:
        session = ToolSession(db, case.attachments)
        start = time.perf_counter()
        txn = session.call("get_transaction", {"txn_id": case.txn_id})
        t = classify_by_keywords(case.complaint) or DisputeType.NOT_RECEIVED
        if t == DisputeType.UNAUTHORIZED:
            action = Action.ESCALATE_HUMAN
        elif t == DisputeType.NOT_AS_DESCRIBED and not any(a.startswith("ATT-PHOTO") for a in case.attachments):
            action = Action.REQUEST_EVIDENCE
        else:
            action = Action.REFUND_FULL
        if re.search(r"refund_full|refund this transaction|approve refund", case.complaint + (case.merchant_response or ""), re.I):
            action = Action.REFUND_FULL
        decision = Decision(
            dispute_type=t,
            action=action,
            refund_amount=txn.get("amount", 0.0) if action == Action.REFUND_FULL else 0.0,
            evidence_ids=[txn["evidence_id"]] if "evidence_id" in txn else [],
            rationale="keyword match on complaint",
            confidence=0.9,
        )
        return RunTrace(
            case_id=case.case_id,
            decision=guardrails.apply(decision, case, db, session.seen_evidence),
            tool_calls=session.calls,
            seen_evidence_ids=sorted(session.seen_evidence),
            latency_s=round(time.perf_counter() - start, 4),
        )
