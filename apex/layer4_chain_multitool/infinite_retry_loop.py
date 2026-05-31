"""
apex/layer4_chain_multitool/infinite_retry_loop.py
--------------------------------------------------
APEX Layer 4 — Chain & Multi-Tool
Failure Mode 4.3: Infinite Retry Loop

Definition:
    A tool repeatedly returns an error or unsatisfactory result.
    The agent retries the same call without backoff, parameter variation,
    or an exit condition — consuming budget and blocking progress.

Detection difficulty: MEDIUM
    Detectable via call count metrics or timeout. The looping behaviour
    itself may not surface in the final response.

TODO: implement EvalModule subclass InfiniteRetryLoopEval
"""
