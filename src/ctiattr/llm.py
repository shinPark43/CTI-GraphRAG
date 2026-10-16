"""OpenAI-compatible chat client for the answering model (Ollama / vLLM).

Every call returns a `Call` record so runs keep the full prompt, output, token
usage and latency for later error analysis.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from openai import OpenAI

from ctiattr.config import LLMConfig


@dataclass
class Call:
    system: str | None
    prompt: str
    output: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float

    def to_dict(self) -> dict:
        return asdict(self)


class ChatLLM:
    def __init__(self, cfg: LLMConfig, *, temperature: float = 0.0, max_retries: int = 4,
                 timeout: float = 600.0):
        self.cfg = cfg
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=timeout)

    @property
    def model(self) -> str:
        return self.cfg.model

    def chat(self, prompt: str, *, system: str | None = None, max_tokens: int = 512) -> Call:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        last: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model, messages=messages,
                    temperature=self.temperature, max_tokens=max_tokens,
                )
            except Exception as e:  # noqa: BLE001 - retry any transport/server error
                last = e
                time.sleep(2 ** attempt)
                continue
            usage = resp.usage
            return Call(
                system=system, prompt=prompt,
                output=resp.choices[0].message.content or "",
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                latency_s=round(time.perf_counter() - t0, 3),
            )
        raise RuntimeError(f"chat failed after {self.max_retries} retries: {last}")
