"""Regression guard: a bad merge once duplicated 11 lines in orchestrator.py,
including `guardian.note_entry()` firing TWICE per entry (double-counting the
per-hour entry cap and throttling learning throughput) and a second copy of the
whole `_audit` method. Guard against consecutive exact-duplicate statement lines
sneaking back into the hot decision path.
"""
import os

ORCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "app", "orchestrator.py")

# lines that are legitimately allowed to repeat back-to-back (closers, blanks)
_ALLOWED = {")", "(", "]", "}", "):", "else:", "try:", "continue", "pass",
            "break", "return"}


def test_no_consecutive_duplicate_statements():
    with open(ORCH) as fh:
        lines = fh.read().split("\n")
    dups = []
    prev = None
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if s and s == prev and s not in _ALLOWED and not s.startswith("#"):
            dups.append((i, s))
        prev = s
    assert not dups, "duplicate consecutive lines in orchestrator.py: " + \
        "; ".join(f"L{n}:{t[:60]}" for n, t in dups)


def test_single_audit_method():
    with open(ORCH) as fh:
        src = fh.read()
    assert src.count("def _audit(") == 1


def test_single_guardian_note_entry_per_entry_block():
    """note_entry must appear once per entry path (limit-order placement +
    market conviction + explore = 3), never doubled within a block."""
    with open(ORCH) as fh:
        lines = fh.read().split("\n")
    note_lines = [i for i, l in enumerate(lines) if "guardian.note_entry()" in l]
    # exactly three call sites, and no two adjacent
    assert len(note_lines) == 3
    assert all(b - a > 1 for a, b in zip(note_lines, note_lines[1:]))
