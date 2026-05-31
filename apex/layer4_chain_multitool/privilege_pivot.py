"""
apex/layer4_chain_multitool/privilege_pivot.py
----------------------------------------------
APEX Layer 4 — Chain & Multi-Tool
Failure Mode 4.2: Privilege Pivot

Definition:
    The agent uses the output of a low-privilege tool call as a credential
    or token to access a higher-privilege tool, crossing an authorization
    boundary it was not supposed to cross.

Detection difficulty: HIGH — SILENT
    Both tool calls may succeed individually. The boundary violation is
    only apparent when examining the cross-tool data flow and permissions.

TODO: implement EvalModule subclass PrivilegePivotEval
"""
