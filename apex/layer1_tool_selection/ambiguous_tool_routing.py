"""
apex/layer1_tool_selection/ambiguous_tool_routing.py
----------------------------------------------------
APEX Layer 1 — Tool Selection
Failure Mode 1.4: Ambiguous Tool Routing

Definition:
    The user's request maps plausibly to multiple tools. The agent
    picks one without resolving the ambiguity, leading to a partially
    correct or contextually wrong result.

Detection difficulty: MEDIUM-HIGH
    The chosen tool may return valid data, but for the wrong semantic
    scope. Requires intent-level analysis to detect.

TODO: implement EvalModule subclass AmbiguousToolRoutingEval
"""
