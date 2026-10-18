"""Report-layer build: clause chunking and the extraction checks (CTIFoundry Steps 0 and 2)."""

import pytest
from pydantic import ValidationError

from ctiattr.report_layer import Chunk, chunk_report, reply_schema, validate
from ctiattr.schemas import ExtractedTTP, GroundedBehavior

TEXT = ("Since November 2021 the group has operated a ransomware service. "
        "Affiliates gain access with stolen VPN credentials; they then disable security tools. "
        "Files are encrypted with a .cuba extension. " * 6)


def test_chunks_cover_text_with_exact_offsets():
    chunks = chunk_report("BLOG-1", TEXT)
    assert chunks, "no chunks produced"
    for c in chunks:
        assert TEXT[c.start:c.end] == c.text
        assert c.chunk_id.startswith("BLOG-1#")
    # consecutive, non-overlapping, in order
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:]))


def test_chunks_respect_size_bounds_and_clause_boundaries():
    for c in chunk_report("BLOG-1", TEXT):
        assert len(c.text) <= 900
        assert c.text.rstrip()[-1] in ".;!?"


def test_single_short_report_is_one_chunk():
    chunks = chunk_report("BLOG-2", "A short summary.")
    assert [c.text for c in chunks] == ["A short summary."]


CHUNK = Chunk("BLOG-1#0", "BLOG-1", 0, 60, "Affiliates use stolen VPN credentials and encrypt files.")


def test_validate_keeps_verbatim_known_ids():
    kept, why = validate(CHUNK, [{"behavior": "encrypt files", "technique_id": "t1486",
                                  "technique_name": "Data Encrypted for Impact"}], {"T1486"})
    assert [(k.technique_id, k.behavior) for k in kept] == [("T1486", "encrypt files")]
    assert why == {"schema": 0, "unknown_id": 0, "not_verbatim": 0}


def test_validate_drops_unknown_ids_and_paraphrases():
    raw = [{"behavior": "encrypt files", "technique_id": "T9999"},          # not in the snapshot
           {"behavior": "encrypts the files", "technique_id": "T1486"},     # not verbatim
           {"behavior": "", "technique_id": "T1486"},                       # empty: schema
           {"behavior": "encrypt files", "technique_id": "1486"}]           # bad ID format: schema
    for r in raw:
        r["technique_name"] = ""
    kept, why = validate(CHUNK, raw, {"T1486"})
    assert kept == []
    assert why == {"schema": 2, "unknown_id": 1, "not_verbatim": 1}


def test_validate_tolerates_malformed_records():
    kept, why = validate(CHUNK, [{}, {"technique_id": None}, "not a dict"], {"T1486"})
    assert kept == [] and why["schema"] == 3


def test_reply_schema_is_strict():
    s = reply_schema()
    item = s["properties"]["ttps"]["items"]
    assert s["additionalProperties"] is False and s["required"] == ["ttps"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"behavior", "technique_id", "technique_name"}
    assert "$defs" not in s and "$ref" not in str(s)


def test_extracted_ttp_normalizes_and_rejects():
    t = ExtractedTTP.model_validate({"behavior": " encrypt\n files ", "technique_id": " t1486 ",
                                     "technique_name": "Data Encrypted for Impact"})
    assert (t.behavior, t.technique_id) == ("encrypt files", "T1486")
    with pytest.raises(ValidationError):
        ExtractedTTP.model_validate({"behavior": "x", "technique_id": "T1486", "technique_name": "", "tactic": "y"})


def test_grounding_errors_are_typed():
    rec = {"chunk_id": "BLOG-1#0", "report_id": "BLOG-1", "behavior": "encrypt files", "technique_id": "T1490"}
    with pytest.raises(ValidationError) as e:
        GroundedBehavior.model_validate(rec, context={"chunk_text": CHUNK.text, "technique_ids": {"T1486"}})
    assert e.value.errors()[0]["type"] == "unknown_id"
    GroundedBehavior.model_validate(rec)  # without context only the shape is checked
