"""
apex/layer3_output_consumption/stale_data_trust.py
--------------------------------------------------
APEX Layer 3 — Output Consumption
Failure Mode 3.2: Stale Data Trust

Definition:
    The tool returns data that is technically valid but outdated —
    cached, replicated with lag, or from a superseded record.
    The agent presents it as current without acknowledging staleness.

Detection difficulty: HIGH — SILENT
    No error is raised. The data looks correct. Only detectable by
    comparing the result's timestamp or version against ground truth.

TODO: implement EvalModule subclass StaleDataTrustEval
"""
