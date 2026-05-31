"""
apex/layer1_tool_selection/tool_omission.py
-------------------------------------------
APEX Layer 1 — Tool Selection
Failure Mode 1.2: Tool Omission

Definition:
    The agent answers the user's question from memory or reasoning
    instead of calling the required tool. The response may sound
    plausible but is not grounded in live data.

Detection difficulty: HIGH — SILENT
    No error is thrown. The agent confidently returns a response.
    Only detectable by comparing against a live tool result.

TODO: implement EvalModule subclass ToolOmissionEval
"""
