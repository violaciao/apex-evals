"""
apex/layer4_chain_multitool/state_corruption.py
-----------------------------------------------
APEX Layer 4 — Chain & Multi-Tool
Failure Mode 4.4: State Corruption

Definition:
    An earlier tool call mutates shared state (database record, session,
    file) in a way that causes a later tool call to operate on invalid
    or inconsistent data — producing wrong results or breaking invariants.

Detection difficulty: HIGH — SILENT
    Each tool call may return success. The corruption is only visible
    when the full state sequence is inspected end-to-end.

TODO: implement EvalModule subclass StateCorruptionEval
"""
