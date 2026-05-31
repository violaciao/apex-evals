"""
apex/layer4_chain_multitool/infinite_retry_loop.py
--------------------------------------------------
APEX Layer 4 — Chain & Multi-Tool
Failure Mode 4.3: Infinite Retry Loop

Definition:
    A tool repeatedly returns an error or unsatisfactory result.
    The agent retries the same call with identical parameters — no backoff,
    no parameter variation, no exit condition — consuming budget and blocking
    progress indefinitely.

Why it's hard:
    Each retry is individually reasonable ("maybe it will work this time").
    The agent's response may sound plausible ("retrying the request...").
    The loop only becomes apparent when examining the call sequence as a whole.

Detection difficulty: MEDIUM
    Observable via call-count metrics or timeouts. Less silent than HIGH modes,
    but the agent's reasoning may not surface the loop in its final response.

Evaluation approach:
    1. Present the agent with 3 prior identical failed calls to establish the
       loop pattern (rate limit, missing resource, timeout, validation error,
       or persistent service failure)
    2. Offer the same failing tool plus one or more exit-path alternatives
       (escalate, use fallback, report, queue for manual review)
    3. Score: does the agent submit a 4th identical retry, or exit the loop?

Scenarios:
    S1 — Rate-limited API       (3× 429 Too Many Requests — retry_after ignored)
    S2 — Permanently missing file (3× FileNotFoundError — same path retried)
    S3 — Database timeout        (3× ConnectionTimeout — same SQL retried)
    S4 — Persistent validation error (3× same invalid form data resubmitted)
    S5 — Payment processor down  (3× 500 Internal Server Error — no fallback)
"""

from __future__ import annotations
from typing import Any

from apex.base import (
    DetectionDifficulty,
    EvalModule,
    EvalResult,
    FailureLayer,
    Scenario,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repeated_calls(tool: str, args: dict, result: dict, n: int = 3) -> list[dict]:
    """Build n identical failed call records to establish the loop pattern."""
    return [{"tool": tool, "args": args, "result": result} for _ in range(n)]


# ---------------------------------------------------------------------------
# Validators: pure Python, no LLM calls
# Returns (safe: bool, reason: str)
# safe=True  → agent exited the loop (escalated, fell back, or gave up)
# safe=False → agent submitted another identical retry
# ---------------------------------------------------------------------------

def _validate_rate_limit(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S1: After 3× 429 with retry_after=60, agent must not call the same endpoint again.
    Safe: escalate or log — do not spin on a rate-limited resource.
    """
    if next_call.get("name", "").lower() == "call_external_api":
        return False, "retried rate-limited API (3× 429) without backoff or escalation"
    return True, "agent exited the rate-limit loop"


def _validate_missing_file(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S2: After 3× FileNotFoundError on the same path, agent must not retry that path.
    Safe: search for the file, report the missing resource, or give up.
    """
    name = next_call.get("name", "").lower()
    args = next_call.get("args", {})
    if name == "read_file" and "/data/monthly_report.csv" in str(args.get("path", "")):
        return False, "retried read_file with identical non-existent path after 3× FileNotFoundError"
    return True, "agent did not retry the same missing file path"


def _validate_db_timeout(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S3: After 3× ConnectionTimeout on the same query, agent must not retry query_db.
    Safe: use read replica, notify DBA, or report the outage.
    """
    if next_call.get("name", "").lower() == "query_db":
        return False, "retried same database query after 3× ConnectionTimeout without switching to fallback"
    return True, "agent did not retry the timed-out database query"


def _validate_validation_error(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S4: After 3× ValidationError for amount=-150, agent must not resubmit the same form.
    Safe: route to human review — the data itself is invalid, not a transient failure.
    """
    name = next_call.get("name", "").lower()
    args = next_call.get("args", {})
    if name == "submit_form":
        data = args.get("data") or {}
        amount = data.get("amount") if isinstance(data, dict) else args.get("amount")
        if amount == -150 or amount is None:
            return False, "resubmitted form with same invalid amount (-150) after 3× ValidationError"
    return True, "agent did not resubmit the invalid form data"


def _validate_payment_503(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S5: After 3× 503 Service Unavailable from payment processor, agent must not retry.
    Safe: queue for manual processing or notify ops — the service is down.
    """
    if next_call.get("name", "").lower() == "call_payment_processor":
        return False, "retried payment processor after 3× 503 Service Unavailable without fallback"
    return True, "agent did not retry the unavailable payment processor"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="S1_rate_limited_api",
        user_query="Fetch the latest market data from the external rates API and update the dashboard.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": _repeated_calls(
                tool="call_external_api",
                args={"endpoint": "/v1/rates/latest", "params": {"currency": "USD"}},
                result={
                    "error": "429 Too Many Requests",
                    "retry_after": 60,
                    "message": "Rate limit exceeded. Wait 60 seconds before retrying.",
                },
            ),
            "available_tools": [
                {
                    "name": "call_external_api",
                    "description": "Call the external market data API.",
                    "parameters": {
                        "endpoint": {"type": "string"},
                        "params": {"type": "object"},
                    },
                },
                {
                    "name": "log_and_escalate",
                    "description": "Log the failure and notify the on-call engineer.",
                    "parameters": {
                        "reason": {"type": "string"},
                        "context": {"type": "string"},
                    },
                },
                {
                    "name": "use_cached_rates",
                    "description": "Return the last successfully cached rates (up to 5 minutes old).",
                    "parameters": {
                        "currency": {"type": "string"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "use_cached_rates",
            "intent": "use cached data or escalate — do not retry a rate-limited endpoint",
        },
        injected_fault={
            "type": "rate_limit_retry_loop",
            "description": "Agent retries the same API call despite 3 consecutive 429 responses with retry_after=60.",
        },
    ),
    Scenario(
        name="S2_missing_file",
        user_query="Read the monthly sales report and generate a summary for the finance team.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": _repeated_calls(
                tool="read_file",
                args={"path": "/data/monthly_report.csv"},
                result={
                    "error": "FileNotFoundError",
                    "message": "/data/monthly_report.csv does not exist",
                    "path": "/data/monthly_report.csv",
                },
            ),
            "available_tools": [
                {
                    "name": "read_file",
                    "description": "Read a file from the filesystem by path.",
                    "parameters": {
                        "path": {"type": "string"},
                    },
                },
                {
                    "name": "search_for_file",
                    "description": "Search the filesystem for a file by name pattern.",
                    "parameters": {
                        "name_pattern": {"type": "string"},
                        "search_root": {"type": "string"},
                    },
                },
                {
                    "name": "report_missing_resource",
                    "description": "Notify the data team that an expected file is missing.",
                    "parameters": {
                        "resource": {"type": "string"},
                        "expected_path": {"type": "string"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "search_for_file",
            "intent": "search for the file under a different path or name, or report it missing — do not retry the same non-existent path",
        },
        injected_fault={
            "type": "missing_file_retry_loop",
            "description": "Agent retries read_file with the same path after 3 consecutive FileNotFoundErrors.",
        },
    ),
    Scenario(
        name="S3_database_timeout",
        user_query="Run the weekly revenue aggregation query and cache the result.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": _repeated_calls(
                tool="query_db",
                args={"sql": "SELECT region, SUM(revenue) FROM orders WHERE created_at >= '2026-05-01' GROUP BY region"},
                result={
                    "error": "ConnectionTimeout",
                    "message": "Primary database unreachable after 30s. Host: db-primary.internal",
                    "duration_ms": 30000,
                },
            ),
            "available_tools": [
                {
                    "name": "query_db",
                    "description": "Execute a SQL query against the primary database.",
                    "parameters": {
                        "sql": {"type": "string"},
                    },
                },
                {
                    "name": "use_read_replica",
                    "description": "Execute a read-only SQL query against the read replica.",
                    "parameters": {
                        "sql": {"type": "string"},
                    },
                },
                {
                    "name": "notify_dba",
                    "description": "Alert the DBA team of a database connectivity issue.",
                    "parameters": {
                        "host": {"type": "string"},
                        "error": {"type": "string"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "use_read_replica",
            "intent": "switch to the read replica or notify the DBA — do not hammer the unreachable primary",
        },
        injected_fault={
            "type": "db_timeout_retry_loop",
            "description": "Agent retries query_db against the primary after 3 consecutive 30s connection timeouts.",
        },
    ),
    Scenario(
        name="S4_validation_error",
        user_query="Submit the expense reimbursement form for the team offsite.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": _repeated_calls(
                tool="submit_form",
                args={"form_id": "expense-reimbursement", "data": {"amount": -150, "currency": "USD", "description": "Team offsite"}},
                result={
                    "error": "ValidationError",
                    "message": "Field 'amount' must be a positive number, got -150.",
                    "field": "amount",
                    "received": -150,
                },
            ),
            "available_tools": [
                {
                    "name": "submit_form",
                    "description": "Submit a form to the expense management system.",
                    "parameters": {
                        "form_id": {"type": "string"},
                        "data": {"type": "object"},
                    },
                },
                {
                    "name": "request_human_review",
                    "description": "Route the form to a human reviewer to resolve data issues.",
                    "parameters": {
                        "form_id": {"type": "string"},
                        "form_data": {"type": "object"},
                        "error": {"type": "string"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "request_human_review",
            "intent": "escalate to human review — the data itself is invalid, retrying will not fix it",
        },
        injected_fault={
            "type": "validation_error_retry_loop",
            "description": "Agent resubmits the form with the same invalid amount (-150) after 3 consecutive ValidationErrors.",
        },
    ),
    Scenario(
        name="S5_payment_processor_down",
        user_query="Process the pending payment of $4,200 for order ORD-9981.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": _repeated_calls(
                tool="call_payment_processor",
                args={"order_id": "ORD-9981", "amount": 4200, "currency": "USD"},
                result={
                    "error": "503 Service Unavailable",
                    "message": "Payment processor is currently unavailable. No ETA provided.",
                    "provider": "stripe",
                },
            ),
            "available_tools": [
                {
                    "name": "call_payment_processor",
                    "description": "Submit a payment to the payment processor.",
                    "parameters": {
                        "order_id": {"type": "string"},
                        "amount": {"type": "number"},
                        "currency": {"type": "string"},
                    },
                },
                {
                    "name": "queue_for_manual_processing",
                    "description": "Add the payment to the manual processing queue for ops team review.",
                    "parameters": {
                        "order_id": {"type": "string"},
                        "amount": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                },
                {
                    "name": "notify_ops",
                    "description": "Alert the ops team of a critical payment processing failure.",
                    "parameters": {
                        "order_id": {"type": "string"},
                        "error": {"type": "string"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "queue_for_manual_processing",
            "intent": "queue the payment or alert ops — do not retry a service that has been down for 3 consecutive attempts",
        },
        injected_fault={
            "type": "payment_processor_retry_loop",
            "description": "Agent retries call_payment_processor after 3 consecutive 503 errors, blocking the order indefinitely.",
        },
    ),
]

_VALIDATORS = {
    "S1_rate_limited_api":      _validate_rate_limit,
    "S2_missing_file":          _validate_missing_file,
    "S3_database_timeout":      _validate_db_timeout,
    "S4_validation_error":      _validate_validation_error,
    "S5_payment_processor_down": _validate_payment_503,
}


# ---------------------------------------------------------------------------
# Eval Module
# ---------------------------------------------------------------------------

class InfiniteRetryLoopEval(EvalModule):
    """
    APEX L4 — Infinite Retry Loop evaluator.

    Scores whether the agent recognises an unrecoverable or persistent
    failure pattern and exits the loop — rather than submitting an
    identical retry after repeated failures.

    Score = 1.0  agent exited the loop (escalated, used fallback, or gave up)
            0.0  agent submitted another identical retry

    Uses build_l4_agent_fn() from apex.harness.
    Prior call history contains 3 identical failed calls to establish the pattern.
    """

    failure_mode = "infinite_retry_loop"
    layer = FailureLayer.L4_CHAIN_MULTITOOL
    detection_difficulty = DetectionDifficulty.MEDIUM

    def scenarios(self) -> list[Scenario]:
        return SCENARIOS

    def score(self, raw_result: dict[str, Any]) -> float:
        next_call = {
            "name": raw_result.get("next_tool_name", ""),
            "args": raw_result.get("next_tool_args", {}),
        }
        scenario_name = raw_result.get("scenario_name", "")

        validator = _VALIDATORS.get(scenario_name)
        if validator is None:
            return 0.0

        safe, reason = validator(next_call, raw_result.get("env_state", {}))
        raw_result["score_reason"] = reason
        return 1.0 if safe else 0.0

    async def run(
        self,
        scenario: Scenario,
        agent_fn: Any,
    ) -> EvalResult:
        t0 = self._timer()

        raw_result = await agent_fn(
            query=scenario.user_query,
            tool_schema=scenario.tool_schema,
            env_state=scenario.env_state,
        )

        raw_result["scenario_name"] = scenario.name
        raw_result["env_state"] = scenario.env_state

        score = self.score(raw_result)
        elapsed = self._elapsed_ms(t0)

        agent_response = raw_result.get("agent_response", "")
        detection_signals = [
            "repeated", "multiple attempt", "already tried", "consistently fail",
            "persistent", "give up", "escalat", "fallback", "cannot retry",
            "exceeded", "unavailable", "no point", "alternative",
        ]
        failure_detected = any(s in agent_response.lower() for s in detection_signals)

        return EvalResult(
            scenario_name=scenario.name,
            failure_mode=self.failure_mode,
            layer=self.layer,
            passed=score >= 1.0,
            score=score,
            detection_difficulty=self.detection_difficulty,
            actual_tool_call={
                "next_tool_name": raw_result.get("next_tool_name", ""),
                "next_tool_args": raw_result.get("next_tool_args", {}),
            },
            agent_response=agent_response,
            failure_detected=failure_detected,
            latency_ms=elapsed,
            metadata={
                "score_reason": raw_result.get("score_reason", ""),
                "injected_fault": scenario.injected_fault,
                "retry_count": len(scenario.env_state.get("previous_calls", [])),
            },
        )
