import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dispute_agent import datagen, evaluate, guardrails, policy
from dispute_agent.agents import LLMAgent, NaiveBaseline, classify_by_keywords
from dispute_agent.backend import Backend, ToolSession
from dispute_agent.schemas import Action, Case, Decision, DisputeType


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    return datagen.generate(n=120, seed=3, out_dir=out)


def small_db() -> Backend:
    db = Backend()
    db.transactions = {
        "T1": {"txn_id": "T1", "customer_id": "C1", "merchant": "M", "amount": 100.0, "currency": "MYR",
               "amount_usd": 22.0, "timestamp": "2026-09-01T10:00", "category": "goods"},
        "T2": {"txn_id": "T2", "customer_id": "C1", "merchant": "M", "amount": 100.0, "currency": "MYR",
               "amount_usd": 22.0, "timestamp": "2026-09-01T10:05", "category": "goods"},
        "T3": {"txn_id": "T3", "customer_id": "C1", "merchant": "Big", "amount": 3000.0, "currency": "MYR",
               "amount_usd": 660.0, "timestamp": "2026-09-01T10:00", "category": "goods"},
    }
    return db


def case(txn="T2", **kw) -> Case:
    base = dict(case_id="X", customer_id="C1", txn_id=txn, created_at="2026-09-10", complaint="x")
    return Case(**(base | kw))


def test_duplicate_within_window_is_refunded():
    lab = policy.decide(case(), DisputeType.DUPLICATE_CHARGE, small_db())
    assert lab.action == Action.REFUND_FULL and lab.refund_amount == 100.0


def test_refund_over_limit_escalates():
    db = small_db()
    db.shipments["T3"] = {"status": "lost", "eta": "2026-09-05"}
    lab = policy.decide(case("T3"), DisputeType.NOT_RECEIVED, db)
    assert lab.action == Action.ESCALATE_HUMAN and "POL-07" in lab.policy_ids


def test_abuse_counts_only_90_day_window():
    db = small_db()
    db.shipments["T2"] = {"status": "lost", "eta": "2026-09-05"}
    db.dispute_history["C1"] = [{"opened_at": d} for d in ["2026-08-01", "2026-07-01", "2026-01-01"]]
    assert policy.decide(case(), DisputeType.NOT_RECEIVED, db).action == Action.REFUND_FULL
    db.dispute_history["C1"].append({"opened_at": "2026-09-01"})
    assert policy.decide(case(), DisputeType.NOT_RECEIVED, db).action == Action.ESCALATE_HUMAN


def test_policy_search_finds_relevant_clause():
    ids = [pid for pid, _ in policy.search("customer charged twice duplicate")]
    assert "POL-02" in ids


def test_generation_is_deterministic(tmp_path):
    _, a = datagen.generate(30, seed=11, out_dir=tmp_path / "a")
    _, b = datagen.generate(30, seed=11, out_dir=tmp_path / "b")
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]


def test_injection_only_on_non_refund_cases(dataset):
    _, cases = dataset
    inj = [c for c in cases if "injection" in c.case.tags]
    assert inj and all(c.label.action != Action.REFUND_FULL for c in inj)


def decision(**kw) -> Decision:
    base = dict(dispute_type="duplicate_charge", action="refund_full", refund_amount=100.0,
                evidence_ids=["EV-TXN-T2"], confidence=0.9)
    return Decision(**(base | kw))


def test_guardrail_passes_clean_decision():
    g = guardrails.apply(decision(), case(), small_db(), {"EV-TXN-T2"})
    assert g.final.action == Action.REFUND_FULL and not g.guardrails_triggered


@pytest.mark.parametrize("kw,seen,expected", [
    ({"evidence_ids": ["EV-MADE-UP"]}, {"EV-TXN-T2"}, "G4_fabricated_evidence"),
    ({"evidence_ids": []}, set(), "G4_no_evidence_cited"),
    ({"confidence": 0.3}, {"EV-TXN-T2"}, "G3_low_confidence"),
    ({"refund_amount": 500.0}, {"EV-TXN-T2"}, "G1_refund_exceeds_transaction"),
])
def test_guardrail_blocks(kw, seen, expected):
    g = guardrails.apply(decision(**kw), case(), small_db(), seen)
    assert g.final.action == Action.ESCALATE_HUMAN and expected in g.guardrails_triggered


def test_guardrail_blocks_double_refund_and_large_refund():
    db = small_db()
    db.refunds["T2"] = [{"refund_id": "R", "amount": 100.0}]
    assert "G5_already_refunded" in guardrails.apply(decision(), case(), db, {"EV-TXN-T2"}).guardrails_triggered
    g = guardrails.apply(decision(refund_amount=3000.0, evidence_ids=["EV-TXN-T3"]), case("T3"), small_db(), {"EV-TXN-T3"})
    assert "G2_over_auto_refund_limit" in g.guardrails_triggered


def test_tool_session_records_evidence():
    s = ToolSession(small_db(), ["ATT-PHOTO-1"])
    out = s.call("list_customer_transactions", {"customer_id": "C1", "merchant": "m"})
    assert len(out["transactions"]) == 2
    assert out["evidence_id"] in s.seen_evidence and "ATT-PHOTO-1" in s.seen_evidence
    assert "error" in s.call("nope", {})


def block(id_, name, inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


class OracleClient:

    def __init__(self, gold_by_txn):
        self.gold = gold_by_txn
        self.messages = SimpleNamespace(create=self.create)

    def create(self, *, messages, **_):
        usage = SimpleNamespace(input_tokens=100, output_tokens=20)
        txn = messages[0]["content"].split("txn_id: ")[1].split("\n")[0]
        cust = messages[0]["content"].split("customer_id: ")[1].split("\n")[0]
        if len(messages) == 1:
            return SimpleNamespace(usage=usage, content=[
                block("a", "get_transaction", {"txn_id": txn}),
                block("b", "get_refunds", {"txn_id": txn}),
                block("c", "get_dispute_history", {"customer_id": cust}),
            ])
        evidence = [json.loads(r["content"])["evidence_id"] for r in messages[-1]["content"]]
        lab = self.gold[txn]
        return SimpleNamespace(usage=usage, content=[block("d", "submit_decision", {
            "dispute_type": lab.dispute_type.value, "action": lab.action.value,
            "refund_amount": lab.refund_amount, "policy_ids": lab.policy_ids,
            "evidence_ids": evidence, "rationale": "oracle", "confidence": 0.95,
        })])


def test_oracle_agent_scores_perfectly(dataset):
    db, cases = dataset
    client = OracleClient({c.case.txn_id: c.label for c in cases})
    agent = LLMAgent(client=client, prompt_version="v2")
    traces = evaluate.run(agent, db, cases, workers=2)
    r = evaluate.score(cases, traces, db)
    assert r["action_accuracy"] == 1.0
    assert r["unsafe_refund_rate"] == 0.0
    assert r["citation_validity"] == 1.0
    assert r["tokens"]["input"] == 200 * len(cases)


def test_naive_baseline_is_unsafe(dataset):
    db, cases = dataset
    r = evaluate.score(cases, evaluate.run(NaiveBaseline(), db, cases), db)
    assert r["unsafe_refund_rate"] > 0.2
    assert r["injection_resistance"] < 1.0
    md = evaluate.to_markdown({"naive": r})
    assert "Unsafe refund rate" in md


def test_keyword_classifier():
    assert classify_by_keywords("I never made this payment") == DisputeType.UNAUTHORIZED
    assert classify_by_keywords("I was charged twice") == DisputeType.DUPLICATE_CHARGE


def test_prompts_render_policy():
    from dispute_agent.agents import load_prompt
    assert "POL-09" in load_prompt("v2") and "{policy}" not in load_prompt("v2")
    assert Path("dispute_agent/prompts/v1.md").exists()
