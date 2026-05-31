"""
apex/layer2_input_construction/over_under_scoped_query.py
---------------------------------------------------------
APEX Layer 2 — Input Construction
Failure Mode 2.5: Over/Under-Scoped Query

Definition:
    The agent constructs a query that is too broad (returns far more data
    than needed, risking exposure or downstream confusion) or too narrow
    (misses records that satisfy the user's intent).

Detection difficulty: MEDIUM
    The tool succeeds and returns data. Intent-level comparison is
    required to determine whether the scope was correct.

TODO: implement EvalModule subclass OverUnderScopedQueryEval
"""
