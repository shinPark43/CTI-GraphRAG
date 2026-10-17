"""Report-layer build: clause chunking and the extraction checks (CTIFoundry Steps 0 and 2)."""

from ctiattr.report_layer import Chunk, chunk_report, validate

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
    assert why == {"unknown_id": 0, "not_verbatim": 0}


def test_validate_drops_unknown_ids_and_paraphrases():
    raw = [{"behavior": "encrypt files", "technique_id": "T9999"},          # not in the snapshot
           {"behavior": "encrypts the files", "technique_id": "T1486"},     # not verbatim
           {"behavior": "", "technique_id": "T1486"}]                        # empty
    kept, why = validate(CHUNK, raw, {"T1486"})
    assert kept == []
    assert why == {"unknown_id": 1, "not_verbatim": 2}


def test_validate_tolerates_malformed_records():
    kept, _ = validate(CHUNK, [{}, {"technique_id": None}], {"T1486"})
    assert kept == []
