"""
apex/layer3_output_consumption/format_misinterpretation.py
----------------------------------------------------------
APEX Layer 3 — Output Consumption
Failure Mode 3.3: Misinterpretation of Format

Definition:
    The agent misreads the structure of a valid tool result —
    e.g. treating a Unix timestamp as a date string, misreading
    a nested array, confusing null with zero, or inverting a boolean.

Detection difficulty: MEDIUM
    The agent's response diverges from the tool result in a detectable
    way, but the tool call itself was correct.

TODO: implement EvalModule subclass FormatMisinterpretationEval
"""
