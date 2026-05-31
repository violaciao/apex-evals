"""
apex/layer1_tool_selection/false_tool_trigger.py
-------------------------------------------------
APEX Layer 1 — Tool Selection
Failure Mode 1.1: False Tool Trigger

Definition:
    The agent invokes a tool when no tool call was necessary.
    The user's request was answerable from context or general knowledge,
    but the agent incurs latency, cost, and potential side effects by
    calling a tool anyway.

Detection difficulty: MEDIUM
    Observable via tool call logs, but silent in the final response.

TODO: implement EvalModule subclass FalseToolTriggerEval
"""
