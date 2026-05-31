"""
apex/layer3_output_consumption/overconfident_trust.py
-----------------------------------------------------
APEX Layer 3 — Output Consumption
Failure Mode 3.5: Overconfident Trust

Definition:
    The tool returns a low-confidence, estimated, or probabilistic result
    (e.g. a prediction score, a fuzzy match, an ML inference).
    The agent presents it as definitive fact without hedging.

Detection difficulty: HIGH — SILENT
    The tool call and result are both valid. The error is in the agent's
    epistemic framing of the output, not in the data itself.

TODO: implement EvalModule subclass OverconfidentTrustEval
"""
