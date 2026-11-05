from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import datagen, evaluate
from .agents import DEFAULT_MODEL, LLMAgent, NaiveBaseline


def make_agent(name: str, model: str):
    if name == "naive":
        return NaiveBaseline()
    if name.startswith("llm-"):
        return LLMAgent(model=model, prompt_version=name.removeprefix("llm-"))
    raise SystemExit(f"unknown agent {name!r}; use naive, llm-v1 or llm-v2")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dispute-agent")
    p.add_argument("--data", type=Path, default=Path("data"))
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create the synthetic dataset")
    g.add_argument("--n", type=int, default=240)
    g.add_argument("--seed", type=int, default=7)

    e = sub.add_parser("eval", help="run agents and write a report")
    e.add_argument("--agents", nargs="+", default=["naive", "llm-v2"])
    e.add_argument("--limit", type=int, default=None, help="only the first N cases (saves API cost)")
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--model", default=DEFAULT_MODEL)
    e.add_argument("--out", type=Path, default=Path("reports"))

    d = sub.add_parser("demo", help="run one case and print the full trace")
    d.add_argument("case_id")
    d.add_argument("--agent", default="llm-v2")
    d.add_argument("--model", default=DEFAULT_MODEL)

    a = p.parse_args(argv)

    if a.cmd == "generate":
        _, cases = datagen.generate(a.n, a.seed, a.data)
        print(json.dumps(datagen.summarize(cases), indent=2))
        return

    db, cases = datagen.load(a.data)

    if a.cmd == "eval":
        cases = cases[: a.limit] if a.limit else cases
        results = {}
        for name in a.agents:
            agent = make_agent(name, a.model)
            print(f"running {name} on {len(cases)} cases...", flush=True)
            traces = evaluate.run(agent, db, cases, a.workers)
            results[name] = evaluate.score(cases, traces, db)
            evaluate.save(a.out, name, traces, results[name])
        report = evaluate.to_markdown(results)
        (a.out / "report.md").write_text(report, encoding="utf-8")
        print(report)
        return

    if a.cmd == "demo":
        lc = next((c for c in cases if c.case.case_id == a.case_id), None)
        if lc is None:
            raise SystemExit(f"{a.case_id} not found")
        trace = make_agent(a.agent, a.model).run(lc.case, db)
        print("TICKET:", lc.case.model_dump_json(indent=1))
        print("\nTOOL CALLS:")
        for c in trace.tool_calls:
            print("  ", c["tool"], c["args"])
        print("\nDECISION:", trace.decision.model_dump_json(indent=1) if trace.decision else trace.error)
        print("\nPOLICY LABEL:", lc.label.model_dump_json())


if __name__ == "__main__":
    main()
