"""
apex/layer2_input_construction/arg_injection.py
-----------------------------------------------
APEX Layer 2 — Input Construction
Failure Mode 2.3: Argument Injection (CVE-2025-68144)

Definition:
    User-supplied input is embedded into tool arguments without
    sanitization. A crafted input hijacks the tool call — e.g.
    SQL injection, path traversal, or shell metacharacter injection.

Detection difficulty: HIGH — SILENT
    The tool may execute successfully with the injected payload.
    No exception is raised; the breach is in what was executed.

TODO: implement EvalModule subclass ArgInjectionEval
"""
