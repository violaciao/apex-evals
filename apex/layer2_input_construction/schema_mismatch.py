"""
apex/layer2_input_construction/schema_mismatch.py
-------------------------------------------------
APEX Layer 2 — Input Construction
Failure Mode 2.4: Schema Mismatch

Definition:
    The agent uses the wrong field names, outdated schema, or incorrect
    parameter structure when constructing a tool call. The tool may
    silently ignore unknown fields or return empty/wrong results.

Detection difficulty: MEDIUM
    May produce an error (if strict validation) or silent wrong results
    (if the tool ignores unrecognised fields).

TODO: implement EvalModule subclass SchemaMismatchEval
"""
