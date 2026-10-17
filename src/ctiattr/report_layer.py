"""Report layer: ATT&CK behaviors grounded in CTIConnect's report summaries.

Follows CTIFoundry's build (Cheng et al., 2026, §3.2 and App. A/G.3):
  Step 0  chunk each report at clause boundaries into 200-900 characters,
          keeping exact character offsets into the source text;
  Step 2  extract attacker behaviors -> ATT&CK IDs with CTIFoundry's TTP prompt;
  checks  keep a record only if its ID exists in the snapshot and its behavior
          phrase occurs verbatim in its chunk (zero fabrication by construction).

The output is used as labelled examples ("this report wording -> T-id") on ATA
evidence cards, never to answer a question from its own source report.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from openai import OpenAI

from ctiattr.config import RAW

# CTIFoundry App. G.3, verbatim except "The 14 ATT&CK tactics" -> "The ATT&CK tactics":
# the tactic list is filled from the snapshot (Enterprise, Mobile and ICS), so the
# prompt does not assume the Enterprise matrix.
TTP_PROMPT = '''Extract MITRE ATT&CK techniques from this CTI report chunk.
Chunk:
"""{chunk}"""
The ATT&CK tactics (for scope): {tactics}
List every ATTACKER behavior in the chunk that corresponds to a real ATT&CK
technique. For each: "behavior" = the verbatim phrase; "technique_id" = the
ATT&CK id (e.g. T1486 or T1059.001), most specific you are confident in;
"technique_name" = official name.
STRICT rules:
- Attacker techniques ONLY. EXCLUDE defenses/mitigations/recommendations
("enforce MFA", "monitor logs", "apply patches", "network segmentation",
"third-party risk management"), generic outcomes ("data theft", "extortion"),
and bare tool/malware names.
- Include techniques stated concisely as OUTCOMES. Examples:
"encrypts files / systems" -> T1486 Data Encrypted for Impact;
"deletes volume shadow copies / inhibits recovery" -> T1490;
"disables/kills security tools" -> T1562.001 Disable or Modify Tools;
"uses valid/compromised/stolen accounts" -> T1078 Valid Accounts;
"exploits a public-facing app / VPN / web vulnerability" -> T1190;
"spearphishing / phishing email" -> T1566;
"PowerShell / command-line execution" -> T1059.
- Do NOT invent ids. If unsure of the exact id, omit the behavior.
Return JSON: {{"ttps": [{{"behavior": "...", "technique_id": "T....", "technique_name":
"..."}}]}}.'''

_CLAUSE = re.compile(r"(?<=[.!?;])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    report_id: str   # "BLOG-<n>", the same id CTIConnect questions use
    start: int       # character offsets into the report's `preprocessed` text
    end: int
    text: str


@dataclass(frozen=True)
class GroundedBehavior:
    chunk_id: str
    report_id: str
    behavior: str
    technique_id: str
    technique_name: str


def load_reports() -> dict[str, str]:
    out = {}
    with open(RAW / "corpus_reports" / "preprocessed_reports.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[f"BLOG-{r['id']}"] = r.get("preprocessed") or ""
    return out


def chunk_report(report_id: str, text: str, lo: int = 200, hi: int = 900) -> list[Chunk]:
    """Pack whole clauses into chunks of lo-hi characters; never split a clause."""
    spans, pos = [], 0
    for piece in _CLAUSE.split(text):
        if not piece:
            continue
        start = text.index(piece, pos)
        spans.append((start, start + len(piece)))
        pos = start + len(piece)
    chunks, cur = [], None
    for s, e in spans:
        if cur and e - cur[0] > hi:
            chunks.append(cur)
            cur = None
        cur = (cur[0], e) if cur else (s, e)
    if cur:
        if chunks and cur[1] - cur[0] < lo and cur[1] - chunks[-1][0] <= hi:
            chunks[-1] = (chunks[-1][0], cur[1])
        else:
            chunks.append(cur)
    return [Chunk(f"{report_id}#{i}", report_id, s, e, text[s:e]) for i, (s, e) in enumerate(chunks)]


def extract(client: OpenAI, model: str, chunk: Chunk, tactics: list[str]) -> list[dict]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": TTP_PROMPT.format(chunk=chunk.text, tactics=", ".join(tactics))}],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content or "{}").get("ttps") or []
    except json.JSONDecodeError:
        return []


def validate(chunk: Chunk, raw: list[dict], technique_ids: set[str]) -> tuple[list[GroundedBehavior], dict]:
    """Keep records whose ID is in the snapshot and whose behavior is verbatim in the chunk."""
    kept, why = [], {"unknown_id": 0, "not_verbatim": 0}
    squash = lambda t: re.sub(r"\s+", " ", t).strip()
    body = squash(chunk.text)
    for r in raw:
        tid = str(r.get("technique_id", "")).strip().upper()
        beh = squash(str(r.get("behavior", "")))
        if tid not in technique_ids:
            why["unknown_id"] += 1
        elif not beh or beh not in body:
            why["not_verbatim"] += 1
        else:
            kept.append(GroundedBehavior(chunk.chunk_id, chunk.report_id, beh, tid,
                                         str(r.get("technique_name", ""))))
    return kept, why


def to_dict(x) -> dict:
    return asdict(x)
