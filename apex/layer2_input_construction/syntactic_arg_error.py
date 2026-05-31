"""
apex/layer2_input_construction/syntactic_arg_error.py
-----------------------------------------------------
APEX Layer 2 — Input Construction
Failure Mode 2.1: Syntactic Argument Error

Definition:
    The agent constructs a tool call with malformed arguments —
    wrong types, missing required fields, or invalid syntax —
    causing the tool to throw an explicit error.

Detection difficulty: LOW
    The tool raises an exception, making the failure visible.
    Tests whether the agent recovers gracefully or propagates the error.

TODO: implement EvalModule subclass SyntacticArgErrorEval
"""
