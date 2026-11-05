from __future__ import annotations

from .backend import Backend
from .policy import AUTO_REFUND_LIMIT_USD
from .schemas import Action, Case, Decision, GuardedDecision

MIN_CONFIDENCE = 0.6


def apply(decision: Decision, case: Case, db: Backend, seen_evidence: set[str]) -> GuardedDecision:
    triggers: list[str] = []
    txn = db.transactions.get(case.txn_id)
    is_refund = decision.action == Action.REFUND_FULL

    if is_refund:
        if txn is None:
            triggers.append("G1_unknown_transaction")
        else:
            if decision.refund_amount > txn["amount"] + 0.01:
                triggers.append("G1_refund_exceeds_transaction")
            if txn["amount_usd"] > AUTO_REFUND_LIMIT_USD:
                triggers.append("G2_over_auto_refund_limit")
        if db.refunds.get(case.txn_id):
            triggers.append("G5_already_refunded")

    if decision.action != Action.ESCALATE_HUMAN and decision.confidence < MIN_CONFIDENCE:
        triggers.append("G3_low_confidence")

    if decision.action in (Action.REFUND_FULL, Action.DENY):
        if not decision.evidence_ids:
            triggers.append("G4_no_evidence_cited")
        elif any(e not in seen_evidence for e in decision.evidence_ids):
            triggers.append("G4_fabricated_evidence")

    if not triggers:
        return GuardedDecision(raw=decision, final=decision)

    final = decision.model_copy(update={
        "action": Action.ESCALATE_HUMAN,
        "refund_amount": 0.0,
        "rationale": f"[guardrail: {', '.join(triggers)}] {decision.rationale}",
    })
    return GuardedDecision(raw=decision, final=final, guardrails_triggered=triggers)
