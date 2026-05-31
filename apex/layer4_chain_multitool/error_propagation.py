"""
apex/layer4_chain_multitool/error_propagation.py
------------------------------------------------
APEX Layer 4 — Chain & Multi-Tool
Failure Mode 4.1: Error Propagation

Definition:
    A failure in one tool call (wrong result, partial data, or soft error)
    is silently passed as input to the next tool in the chain. The downstream
    tool executes successfully on corrupt input, amplifying the original error.

Detection difficulty: HIGH — SILENT
    Each individual tool call may appear to succeed. The corruption is only
    detectable by tracing the data flow across the full chain.

TODO: implement EvalModule subclass ErrorPropagationEval
"""
