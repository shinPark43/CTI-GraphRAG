"""The explanation post-filter must leave exactly the chosen target ID in the text."""

from ctiattr.sgs import scrub_ids
from ctiattr.vendor.cticonnect_metrics import extract_ids

TITLES = {"T1059": "Command and Scripting Interpreter", "T1059.001": "PowerShell",
          "CWE-79": "Cross-site Scripting", "CAPEC-66": "SQL Injection"}


def test_keeps_chosen_and_replaces_others_with_titles():
    text = "T1059.001 fits; unlike T1059 or CWE-79, it uses PowerShell (see CAPEC-66, CVE-2023-36884)."
    out = scrub_ids(text, "T1059.001", TITLES)
    assert extract_ids(out, "mitre") == {"T1059.001"}
    assert extract_ids(out, "cwe") == set()
    assert "Command and Scripting Interpreter" in out and "SQL Injection" in out
    assert "CVE-2023-36884" in out  # CVEs are never a target type, left as written


def test_unknown_ids_become_generic_words():
    out = scrub_ids("Maybe CWE-99999 instead of CWE-79.", "CWE-79", TITLES)
    assert extract_ids(out, "cwe") == {"CWE-79"}
    assert "a related entry" in out
