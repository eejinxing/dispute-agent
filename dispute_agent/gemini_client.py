from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")


class _Messages:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def create(self, *, model: str, max_tokens: int, system: str, tools: list[dict],
               messages: list[dict], tool_choice=None) -> SimpleNamespace:
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}] + _convert(messages),
            "tools": [
                {"type": "function",
                 "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools
            ],
            "tool_choice": "required",
        }
        data = self._post(body)
        msg = data["choices"][0]["message"]
        content: list[SimpleNamespace] = []
        if msg.get("content"):
            content.append(SimpleNamespace(type="text", text=msg["content"]))
        for i, call in enumerate(msg.get("tool_calls") or []):
            fn = call["function"]
            args = fn.get("arguments") or "{}"
            content.append(SimpleNamespace(type="tool_use", id=call.get("id") or f"call_{len(messages)}_{i}",
                                           name=fn["name"], input=json.loads(args) if isinstance(args, str) else args,
                                           extra=call.get("extra_content")))
        u = data.get("usage") or {}
        usage = SimpleNamespace(input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0))
        return SimpleNamespace(content=content, usage=usage)

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise RuntimeError(f"Gemini error {e.code}: {e.read().decode(errors='replace')[:300]}") from None
        raise RuntimeError("Gemini: too many retries")


class GeminiClient:
    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit('GEMINI_API_KEY is not set. In PowerShell: $env:GEMINI_API_KEY="your key"')
        self.messages = _Messages(key, base_url)


def _convert(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
        elif m["role"] == "assistant":
            text = "".join(b.text for b in c if b.type == "text")
            calls = []
            for b in c:
                if b.type == "tool_use":
                    call = {"id": b.id, "type": "function",
                            "function": {"name": b.name, "arguments": json.dumps(b.input, ensure_ascii=False)}}
                    if getattr(b, "extra", None):
                        call["extra_content"] = b.extra
                    calls.append(call)
            msg: dict = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            for r in c:
                out.append({"role": "tool", "tool_call_id": r["tool_use_id"], "content": r["content"]})
    return out
