"""Structure derivable from the CTIConnect corpus_kb snapshot alone.

No outside data and no construction/ files: everything here is parsed from
corpus_kb/{mitre,cwe,capec,cve}.jsonl.

- ATT&CK domain (Enterprise / Mobile / ICS) from platforms and ID prefix.
- Retired ATT&CK IDs: an Enterprise parent-level ID whose title equals an
  Enterprise sub-technique's title (T1004 "Winlogon Helper DLL" -> T1547.004)
  is the pre-v7 ID that the sub-technique replaced.
- ATT&CK hierarchy from the ID pattern (T1059.001 -> T1059).
- CAPEC records (all have a Description) with their CWE and ATT&CK links.
- CVE descriptions with their CWEs, as labelled examples of each CWE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property

from ctiattr.config import RAW
from ctiattr.data import load_kb

_KB = RAW / "corpus_kb"
_MOBILE = {"Android", "iOS"}


def _jsonl(name: str) -> list[dict]:
    with open(_KB / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _as_list(x) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


@dataclass(frozen=True)
class CapecRecord:
    id: str
    name: str
    description: str
    cwes: tuple[str, ...]
    techniques: tuple[str, ...]
    technique_names: tuple[str, ...] = ()   # CAPEC's own Entry_Name for each ATT&CK link


@dataclass(frozen=True)
class CveRecord:
    id: str
    description: str
    cwes: tuple[str, ...]
    primary: tuple[str, ...] = ()           # CWEs NVD lists as the Primary weakness


@dataclass
class Snapshot:
    mitre: list = field(default_factory=lambda: load_kb("mitre"))
    cwe: list = field(default_factory=lambda: load_kb("cwe"))

    # --- ATT&CK -------------------------------------------------------------

    @cached_property
    def technique_ids(self) -> set[str]:
        return {e.doc_id for e in self.mitre}

    def domain(self, tid: str) -> str:
        e = self._mitre_by_id[tid]
        platforms = set(json.loads(e.text).get("x_mitre_platforms") or [])
        if platforms & _MOBILE:
            return "mobile"
        if tid.startswith("T0"):
            return "ics"
        return "enterprise"

    @cached_property
    def _mitre_by_id(self) -> dict:
        return {e.doc_id: e for e in self.mitre}

    @cached_property
    def retired(self) -> dict[str, str]:
        """Retired parent-level Enterprise ID -> the sub-technique that replaced it."""
        sub_by_title = {}
        for e in self.mitre:
            if "." in e.doc_id and self.domain(e.doc_id) == "enterprise":
                sub_by_title.setdefault(e.title.strip().lower(), e.doc_id)
        return {e.doc_id: sub_by_title[e.title.strip().lower()] for e in self.mitre
                if "." not in e.doc_id and self.domain(e.doc_id) == "enterprise"
                and e.title.strip().lower() in sub_by_title}

    @cached_property
    def answerable_techniques(self) -> set[str]:
        return {t for t in self.technique_ids
                if self.domain(t) == "enterprise" and t not in self.retired}

    @staticmethod
    def parent(tid: str) -> str | None:
        return tid.split(".")[0] if "." in tid else None

    @cached_property
    def children(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for t in sorted(self.technique_ids):
            p = self.parent(t)
            if p:
                out.setdefault(p, []).append(t)
        return out

    def tactics(self, tid: str) -> list[str]:
        phases = json.loads(self._mitre_by_id[tid].text).get("kill_chain_phases") or []
        return [p["phase_name"] for p in phases]

    # --- CWE ----------------------------------------------------------------

    @cached_property
    def cwe_ids(self) -> set[str]:
        return {e.doc_id.upper() for e in self.cwe}

    # --- CAPEC --------------------------------------------------------------

    @cached_property
    def capec(self) -> list[CapecRecord]:
        out = []
        for r in _jsonl("capec"):
            c = json.loads(r["contents"])
            # dict.fromkeys: three records list a link twice (CAPEC-545, -571, -644);
            # count each distinct link once (1,212 CWE links, 271 ATT&CK links)
            rw = (c.get("Related_Weaknesses") or {}).get("Related_Weakness")
            cwes = tuple(dict.fromkeys(f"CWE-{w['@CWE_ID']}" for w in _as_list(rw)
                                       if f"CWE-{w['@CWE_ID']}" in self.cwe_ids))
            tm = (c.get("Taxonomy_Mappings") or {}).get("Taxonomy_Mapping")
            tnames = dict((f"T{m['Entry_ID']}", m.get("Entry_Name") or "") for m in reversed(_as_list(tm))
                          if m.get("@Taxonomy_Name") == "ATTACK" and f"T{m.get('Entry_ID')}" in self.technique_ids)
            techs = tuple(dict.fromkeys(f"T{m['Entry_ID']}" for m in _as_list(tm)
                                        if m.get("@Taxonomy_Name") == "ATTACK"
                                        and f"T{m.get('Entry_ID')}" in self.technique_ids))
            desc = c.get("Description") or ""
            if not isinstance(desc, str):
                desc = json.dumps(desc)
            out.append(CapecRecord(f"CAPEC-{r['capec_id']}", c.get("@Name") or r.get("title", ""),
                                   desc, cwes, techs, tuple(tnames[t] for t in techs)))
        return out

    # --- CVE ----------------------------------------------------------------

    @cached_property
    def cve(self) -> list[CveRecord]:
        out = []
        for r in _jsonl("cve"):
            c = json.loads(r["contents"] or "null") or {}
            desc = next((d["value"] for d in c.get("descriptions") or [] if d.get("value")), "")
            cwes = sorted({d["value"] for w in c.get("weaknesses") or []
                           for d in w.get("description") or []
                           if d.get("value", "").upper() in self.cwe_ids})
            primary = sorted({d["value"] for w in c.get("weaknesses") or [] if w.get("type") == "Primary"
                              for d in w.get("description") or [] if d.get("value", "") in cwes})
            if desc and cwes:
                out.append(CveRecord(r["cve_id"], desc, tuple(cwes), tuple(primary)))
        return out
