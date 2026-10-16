"""QA for the ID-based metrics (Entity Linking + Entity Attribution)."""

from __future__ import annotations

from ctiattr.vendor.cticonnect_metrics import (
    normalize_id, extract_ids, prf1, score_id_item, aggregate_scores,
)


class TestNormalize:
    def test_cwe(self):
        assert normalize_id("cwe 79", "cwe") == "CWE-79"
        assert normalize_id("CWE-79", "cwe") == "CWE-79"
        assert normalize_id("cwe-079", "cwe") == "CWE-079"  # preserves digits

    def test_cve(self):
        assert normalize_id("cve-2024-3400", "cve") == "CVE-2024-3400"
        assert normalize_id("CVE 2024 3400", "cve") == "CVE-2024-3400"

    def test_capec(self):
        assert normalize_id("capec-66", "capec") == "CAPEC-66"

    def test_mitre(self):
        assert normalize_id("t1059", "mitre") == "T1059"
        assert normalize_id("T1059.001", "mitre") == "T1059.001"
        assert normalize_id("technique T1059.001 here", "mitre") == "T1059.001"

    def test_garbage(self):
        assert normalize_id("", "cwe") == ""
        assert normalize_id("no id here", "cwe") == ""


class TestExtract:
    def test_single(self):
        assert extract_ids("- CWE-384.\n- matching CWE-384: ...", "cwe") == {"CWE-384"}

    def test_multiple_distinct(self):
        txt = "could be CWE-79 or CWE-89 depending on context"
        assert extract_ids(txt, "cwe") == {"CWE-79", "CWE-89"}

    def test_mitre_subtechnique(self):
        txt = "1) MITRE ATT&CK Category: Template Injection (T1221)."
        assert extract_ids(txt, "mitre") == {"T1221"}

    def test_mitre_does_not_grab_random_T(self):
        # 'T' inside a word should not match
        assert extract_ids("The TARGET system", "mitre") == set()

    def test_wrong_kind_returns_empty(self):
        assert extract_ids("CWE-79", "cve") == set()


class TestPRF1:
    def test_perfect(self):
        assert prf1({"CWE-79"}, {"CWE-79"}) == (1.0, 1.0, 1.0)

    def test_miss(self):
        assert prf1({"CWE-89"}, {"CWE-79"}) == (0.0, 0.0, 0.0)

    def test_partial_precision(self):
        # predicted 2, 1 correct -> P=.5 R=1 F1=.667
        p, r, f1 = prf1({"CWE-79", "CWE-89"}, {"CWE-79"})
        assert p == 0.5 and r == 1.0
        assert abs(f1 - 0.6667) < 1e-3

    def test_partial_recall(self):
        # gold has 2, predicted 1 correct -> P=1 R=.5
        p, r, f1 = prf1({"T1059"}, {"T1059", "T1003"})
        assert p == 1.0 and r == 0.5

    def test_empty_both(self):
        assert prf1(set(), set()) == (1.0, 1.0, 1.0)

    def test_empty_pred(self):
        assert prf1(set(), {"CWE-79"}) == (0.0, 0.0, 0.0)


class TestScoreItem:
    def test_el_single_correct(self):
        s = score_id_item(
            "- CWE-384.\n- The vulnerability ... matching CWE-384: ...",
            {"target_type": "cwe", "target_id": "CWE-384"},
            task="rcm", item_id="rcm-001",
        )
        assert s.exact_match
        assert s.f1 == 1.0
        assert s.pred_ids == ["CWE-384"]

    def test_el_wrong(self):
        s = score_id_item(
            "- CWE-200.",
            {"target_type": "cwe", "target_id": "CWE-384"},
            task="rcm",
        )
        assert not s.exact_match
        assert s.f1 == 0.0

    def test_ea_set_partial(self):
        # gold {T1059, T1003}; predicted {T1059, T1486}
        s = score_id_item(
            "Techniques: T1059 and T1486",
            {"target_type": "mitre", "target_ids": ["T1059", "T1003"]},
            task="ata",
        )
        assert s.precision == 0.5
        assert s.recall == 0.5
        assert not s.exact_match

    def test_ea_set_perfect(self):
        s = score_id_item(
            "1) CWE-200 ... 2) CWE-732 ...",
            {"target_type": "cwe", "target_ids": ["CWE-200", "CWE-732"]},
            task="vca",
        )
        assert s.exact_match
        assert s.f1 == 1.0


class TestAggregate:
    def test_per_task_and_overall(self):
        items = [
            score_id_item("CWE-79", {"target_type": "cwe", "target_id": "CWE-79"}, task="rcm"),
            score_id_item("CWE-1", {"target_type": "cwe", "target_id": "CWE-2"}, task="rcm"),
            score_id_item("T1059", {"target_type": "mitre", "target_ids": ["T1059"]}, task="ata"),
        ]
        agg = aggregate_scores(items)
        assert agg["overall"]["n"] == 3
        assert agg["per_task"]["rcm"]["n"] == 2
        assert agg["per_task"]["rcm"]["exact_match"] == 0.5  # 1 of 2 correct
        assert agg["per_task"]["ata"]["f1"] == 1.0
