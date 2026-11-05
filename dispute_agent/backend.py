from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import policy


@dataclass
class Backend:
    customers: dict[str, dict] = field(default_factory=dict)
    transactions: dict[str, dict] = field(default_factory=dict)
    shipments: dict[str, dict] = field(default_factory=dict)
    refunds: dict[str, list[dict]] = field(default_factory=dict)
    subscriptions: dict[str, dict] = field(default_factory=dict)
    dispute_history: dict[str, list[dict]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Backend":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_transaction",
        "description": "Look up one transaction: amount, currency, amount_usd, merchant, timestamp, category.",
        "input_schema": {"type": "object", "properties": {"txn_id": {"type": "string"}}, "required": ["txn_id"]},
    },
    {
        "name": "list_customer_transactions",
        "description": "List a customer's transactions, optionally filtered by merchant. Use to check duplicate charges.",
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}, "merchant": {"type": "string"}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "get_shipment",
        "description": "Shipment tracking for a transaction: status (delivered / in_transit / lost), eta, delivered_at.",
        "input_schema": {"type": "object", "properties": {"txn_id": {"type": "string"}}, "required": ["txn_id"]},
    },
    {
        "name": "get_refunds",
        "description": "Refunds already issued for a transaction.",
        "input_schema": {"type": "object", "properties": {"txn_id": {"type": "string"}}, "required": ["txn_id"]},
    },
    {
        "name": "get_subscription",
        "description": "Subscription record for a transaction: plan and cancelled_at date (null if active).",
        "input_schema": {"type": "object", "properties": {"txn_id": {"type": "string"}}, "required": ["txn_id"]},
    },
    {
        "name": "get_dispute_history",
        "description": "Prior disputes opened by this customer, with opened_at dates.",
        "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]},
    },
    {
        "name": "search_policy",
        "description": "Search the dispute policy. Returns matching clauses with ids like POL-03.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
]


class ToolSession:

    def __init__(self, backend: Backend, attachments: list[str] | None = None):
        self.db = backend
        self.calls: list[dict] = []
        self.seen_evidence: set[str] = set(attachments or [])
        self._handlers: dict[str, Callable[..., dict]] = {
            "get_transaction": self.get_transaction,
            "list_customer_transactions": self.list_customer_transactions,
            "get_shipment": self.get_shipment,
            "get_refunds": self.get_refunds,
            "get_subscription": self.get_subscription,
            "get_dispute_history": self.get_dispute_history,
            "search_policy": self.search_policy,
        }

    def call(self, name: str, args: dict) -> dict:
        handler = self._handlers.get(name)
        if handler is None:
            result = {"error": f"unknown tool {name}"}
        else:
            try:
                result = handler(**args)
            except TypeError as e:
                result = {"error": f"bad arguments: {e}"}
        self.calls.append({"tool": name, "args": args, "ok": "error" not in result})
        for key in ("evidence_id",):
            if key in result:
                self.seen_evidence.add(result[key])
        for item in result.get("clauses", []):
            self.seen_evidence.add(item["evidence_id"])
        return result

    def _ev(self, ev: str, payload: dict) -> dict:
        return {"evidence_id": ev, **payload}

    def get_transaction(self, txn_id: str) -> dict:
        t = self.db.transactions.get(txn_id)
        if not t:
            return {"error": "transaction not found"}
        return self._ev(f"EV-TXN-{txn_id}", t)

    def list_customer_transactions(self, customer_id: str, merchant: str | None = None) -> dict:
        rows = [
            t for t in self.db.transactions.values()
            if t["customer_id"] == customer_id and (merchant is None or t["merchant"].lower() == merchant.lower())
        ]
        rows.sort(key=lambda t: t["timestamp"])
        suffix = f"-{merchant}" if merchant else ""
        return self._ev(f"EV-TXNLIST-{customer_id}{suffix}", {"transactions": rows})

    def get_shipment(self, txn_id: str) -> dict:
        s = self.db.shipments.get(txn_id)
        if not s:
            return self._ev(f"EV-SHP-{txn_id}", {"shipment": None, "note": "no shipment record"})
        return self._ev(f"EV-SHP-{txn_id}", {"shipment": s})

    def get_refunds(self, txn_id: str) -> dict:
        return self._ev(f"EV-REF-{txn_id}", {"refunds": self.db.refunds.get(txn_id, [])})

    def get_subscription(self, txn_id: str) -> dict:
        return self._ev(f"EV-SUB-{txn_id}", {"subscription": self.db.subscriptions.get(txn_id)})

    def get_dispute_history(self, customer_id: str) -> dict:
        return self._ev(f"EV-HIST-{customer_id}", {"prior_disputes": self.db.dispute_history.get(customer_id, [])})

    def search_policy(self, query: str) -> dict:
        hits = policy.search(query)
        return {"clauses": [{"evidence_id": f"EV-{pid}", "policy_id": pid, "text": text} for pid, text in hits]}
