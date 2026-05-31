"""
apex/layer1_tool_selection/wrong_tool_selection.py
--------------------------------------------------
APEX Layer 1 — Tool Selection
Failure Mode 1.3: Wrong Tool Selection

Definition:
    The agent selects the wrong tool from the available set.
    The chosen tool executes without error but returns data irrelevant
    to the user's intent (e.g. querying inventory when orders were needed).

Detection difficulty: MEDIUM
    Tool call logs reveal the wrong tool; final response may expose
    the mismatch if the agent uses the result directly.

TODO: implement EvalModule subclass WrongToolSelectionEval
"""
