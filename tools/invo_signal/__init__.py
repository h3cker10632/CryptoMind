"""Invo signal evaluation harness (STANDALONE — not imported by app/).

Answers one question before any Invo data is wired into CryptoMind's learner:
does aggregated Invo top-trader positioning predict forward returns on the
assets we trade, and does it add anything beyond the funding / open-interest /
long-short data we already ingest?

Nothing here executes trades or feeds the live model. It reads *already
collected* snapshots + price series and reports Information Coefficient,
mutual information, orthogonality vs. an existing-feature baseline, and a
walk-forward ablation. The verdict is decided on evidence, exactly like the
scale-out prototype was.
"""
