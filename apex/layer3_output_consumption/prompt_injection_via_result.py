"""
apex/layer3_output_consumption/prompt_injection_via_result.py
-------------------------------------------------------------
APEX Layer 3 — Output Consumption
Failure Mode 3.4: Prompt Injection via Result

Definition:
    A tool result contains embedded natural-language instructions
    (e.g. in a database field, API response body, or file content)
    that the agent interprets and follows as if they were system commands.

Detection difficulty: HIGH — SILENT
    The tool call succeeds normally. The injection is only visible in
    the agent's subsequent behaviour — a changed response, unexpected
    tool call, or policy bypass.

TODO: implement EvalModule subclass PromptInjectionViaResultEval
"""
