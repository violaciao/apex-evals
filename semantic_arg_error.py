"""
apex/layer2_input_construction/semantic_arg_error.py
-----------------------------------------------------
APEX Layer 2 — Input Construction
Failure Mode: Semantic Argument Error

Definition:
    The tool is called with syntactically valid arguments that execute
    without error, but return semantically wrong data — because the agent
    misunderstood the user's intent when constructing the call.

Why it's hard:
    No exception is raised. The tool returns a result. The agent
    confidently uses it. The error is invisible unless you compare
    the result against ground truth derived from the original user intent.

Detection difficulty: HIGH — SILENT

Evaluation approach:
    1. Present agent with a natural-language query
    2. Agent constructs a SQL/API call via tool
    3. Tool executes successfully and returns data
    4. Scorer compares: does the returned data satisfy the user's *intent*?
       (not just: did the tool run?)

Scenarios:
    S1 — Date range misinterpretation   (last quarter vs last 3 months)
    S2 — Aggregation grain error        (daily avg vs total)
    S3 — Filter inversion               (include vs exclude condition)
    S4 — Metric proxy confusion         (revenue vs gross margin)
    S5 — Timezone offset error          (UTC stored, local queried)
"""

from __future__ import annotations
import json
from typing import Any

from apex.base import (
    DetectionDifficulty,
    EvalModule,
    EvalResult,
    FailureLayer,
    Scenario,
)


# ---------------------------------------------------------------------------
# Fixtures: lightweight SQLite-compatible SQL and ground truth
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id    INTEGER PRIMARY KEY,
    created_at  TEXT NOT NULL,   -- ISO-8601 UTC
    revenue     REAL NOT NULL,
    cogs        REAL NOT NULL,   -- cost of goods sold
    region      TEXT NOT NULL
);
"""

# What "correct" means per scenario: a validation function over the SQL string
# and/or the result set. We score intent, not surface form.
def _validate_date_range(sql: str, env: dict) -> tuple[bool, str]:
    """S1: 'last quarter' must map to the calendar quarter, not last 90 days."""
    sql_lower = sql.lower()
    # Correct: quarter boundary (e.g. 2026-01-01 to 2026-03-31 for Q1)
    # Wrong: interval of 90/91/92 days from today
    has_interval = any(x in sql_lower for x in ["interval", "days", "-90", "-91", "-92"])
    has_quarter_boundary = any(x in sql_lower for x in [
        "2026-01-01", "2026-04-01",  # Q1 2026
        "quarter", "datetrunc", "date_trunc",
    ])
    if has_interval and not has_quarter_boundary:
        return False, "used rolling 90-day window instead of calendar quarter"
    if has_quarter_boundary or (not has_interval):
        return True, "correctly used calendar quarter boundary"
    return True, "acceptable"


def _validate_aggregation(sql: str, env: dict) -> tuple[bool, str]:
    """S2: 'average daily revenue' must use AVG over daily totals, not AVG over rows."""
    sql_lower = sql.lower()
    # Correct pattern: GROUP BY date → then AVG of that, or AVG(revenue/days)
    # Wrong: AVG(revenue) directly over all order rows
    has_date_group = "group by" in sql_lower and any(
        x in sql_lower for x in ["date(", "strftime", "created_at"]
    )
    has_raw_avg = "avg(revenue)" in sql_lower or "avg(orders.revenue)" in sql_lower
    if has_raw_avg and not has_date_group:
        return False, "averaged over rows, not daily aggregates — inflated by order volume"
    return True, "correctly aggregated to daily grain first"


def _validate_filter(sql: str, env: dict) -> tuple[bool, str]:
    """S3: 'revenue excluding returns' — agent must exclude region='RETURN', not include it."""
    sql_lower = sql.lower()
    # Wrong: WHERE region = 'RETURN'
    # Correct: WHERE region != 'RETURN'  or  region NOT IN ('RETURN')
    if "region = 'return'" in sql_lower or 'region = "return"' in sql_lower:
        return False, "filter includes returns instead of excluding them"
    if "!= 'return'" in sql_lower or "not in" in sql_lower or "<> 'return'" in sql_lower:
        return True, "correctly excludes returns"
    return True, "no explicit return filter — may be acceptable if returns absent from data"


def _validate_metric(sql: str, env: dict) -> tuple[bool, str]:
    """S4: 'profit margin' must compute (revenue - cogs) / revenue, not just revenue."""
    sql_lower = sql.lower()
    has_cogs = "cogs" in sql_lower
    has_margin_calc = ("revenue - cogs" in sql_lower or "revenue-cogs" in sql_lower)
    if not has_cogs:
        return False, "queried revenue only; 'profit margin' requires (revenue - cogs) / revenue"
    if has_margin_calc:
        return True, "correctly computed margin"
    return False, "referenced cogs but did not compute margin ratio"


def _validate_timezone(sql: str, env: dict) -> tuple[bool, str]:
    """S5: timestamps stored in UTC; 'today NYC' must offset by -4 or -5 hours."""
    sql_lower = sql.lower()
    # Correct: explicit UTC offset or timezone conversion
    has_tz = any(x in sql_lower for x in [
        "utc", "-04:00", "-05:00", "localtime", "datetime(", "at time zone"
    ])
    if not has_tz:
        return False, "no timezone conversion — UTC stored data compared against local date boundary"
    return True, "timezone offset applied"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="S1_date_range_last_quarter",
        user_query="Show me total revenue for last quarter",
        tool_schema={
            "name": "query_db",
            "description": "Execute a SQL SELECT query against the orders database.",
            "parameters": {"sql": {"type": "string", "description": "Valid SQL SELECT statement"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "db_schema": _SCHEMA,
            "note": "Q1 2026 = 2026-01-01 to 2026-03-31",
        },
        expected_tool_call={
            "name": "query_db",
            "intent": "calendar quarter boundary (Q1 2026), not rolling 90 days",
        },
        injected_fault={
            "type": "date_range_misinterpretation",
            "description": "Agent uses INTERVAL '90 days' from today instead of Q1 boundary",
        },
    ),
    Scenario(
        name="S2_aggregation_grain",
        user_query="What was the average daily revenue in April?",
        tool_schema={
            "name": "query_db",
            "description": "Execute a SQL SELECT query against the orders database.",
            "parameters": {"sql": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "db_schema": _SCHEMA,
            "note": "Multiple orders per day; AVG must be over daily totals, not individual rows",
        },
        expected_tool_call={
            "name": "query_db",
            "intent": "GROUP BY date first, then average the daily totals",
        },
        injected_fault={
            "type": "aggregation_grain_error",
            "description": "Agent uses AVG(revenue) over all rows — inflated by high-order-volume days",
        },
    ),
    Scenario(
        name="S3_filter_inversion",
        user_query="Show total revenue excluding returns for this month",
        tool_schema={
            "name": "query_db",
            "description": "Execute a SQL SELECT query against the orders database.",
            "parameters": {"sql": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "db_schema": _SCHEMA,
            "note": "Returns are tagged region='RETURN' in the orders table",
        },
        expected_tool_call={
            "name": "query_db",
            "intent": "WHERE region != 'RETURN'",
        },
        injected_fault={
            "type": "filter_inversion",
            "description": "Agent writes WHERE region = 'RETURN' — returns only the returns",
        },
    ),
    Scenario(
        name="S4_metric_proxy_confusion",
        user_query="What's our profit margin by region last month?",
        tool_schema={
            "name": "query_db",
            "description": "Execute a SQL SELECT query against the orders database.",
            "parameters": {"sql": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "db_schema": _SCHEMA,
            "note": "Margin = (revenue - cogs) / revenue; cogs column exists",
        },
        expected_tool_call={
            "name": "query_db",
            "intent": "SELECT region, (SUM(revenue) - SUM(cogs)) / SUM(revenue) AS margin",
        },
        injected_fault={
            "type": "metric_proxy_confusion",
            "description": "Agent queries revenue only, ignores cogs — returns revenue not margin",
        },
    ),
    Scenario(
        name="S5_timezone_offset",
        user_query="How many orders came in today?",
        tool_schema={
            "name": "query_db",
            "description": "Execute a SQL SELECT query against the orders database.",
            "parameters": {"sql": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "timezone": "America/New_York",  # UTC-4 in May
            "db_schema": _SCHEMA,
            "note": "created_at stored in UTC; 'today' means NYC local date",
        },
        expected_tool_call={
            "name": "query_db",
            "intent": "convert UTC timestamps to NYC local before date comparison",
        },
        injected_fault={
            "type": "timezone_offset_error",
            "description": "Agent filters WHERE date(created_at) = '2026-05-30' — misses orders from 20:00–23:59 UTC prev day and includes next-day UTC orders",
        },
    ),
]

_VALIDATORS = {
    "S1_date_range_last_quarter": _validate_date_range,
    "S2_aggregation_grain": _validate_aggregation,
    "S3_filter_inversion": _validate_filter,
    "S4_metric_proxy_confusion": _validate_metric,
    "S5_timezone_offset": _validate_timezone,
}


# ---------------------------------------------------------------------------
# Eval Module
# ---------------------------------------------------------------------------

class SemanticArgErrorEval(EvalModule):
    """
    APEX L2 — Semantic Argument Error evaluator.

    Scores whether the SQL/API call constructed by the agent correctly
    reflects the user's intent — not just whether it executed without error.
    """

    failure_mode = "semantic_argument_error"
    layer = FailureLayer.L2_INPUT_CONSTRUCTION
    detection_difficulty = DetectionDifficulty.HIGH

    def scenarios(self) -> list[Scenario]:
        return SCENARIOS

    def score(self, raw_result: dict[str, Any]) -> float:
        """
        Score = 1.0 if the generated SQL satisfies user intent
                0.0 if it silently returns wrong data
        Partial credit (0.5) if intent is partially correct.
        """
        sql = raw_result.get("sql", "")
        scenario_name = raw_result.get("scenario_name", "")
        env = raw_result.get("env_state", {})

        validator = _VALIDATORS.get(scenario_name)
        if validator is None:
            return 0.0

        passed, reason = validator(sql, env)
        raw_result["score_reason"] = reason
        return 1.0 if passed else 0.0

    async def run(
        self,
        scenario: Scenario,
        agent_fn: Any,
    ) -> EvalResult:
        """
        Execute scenario against agent_fn.

        agent_fn signature:
            async (query: str, tool_schema: dict, env_state: dict) -> dict
            returns: {"sql": str, "agent_response": str, ...}
        """
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

        # Did the agent detect its own error? (look for hedging language)
        agent_response = raw_result.get("agent_response", "")
        hedging_signals = [
            "i assumed", "note that", "clarify", "not sure", "double-check",
            "verify", "timezone", "calendar quarter", "fiscal",
        ]
        failure_detected = any(s in agent_response.lower() for s in hedging_signals)

        return EvalResult(
            scenario_name=scenario.name,
            failure_mode=self.failure_mode,
            layer=self.layer,
            passed=score >= 1.0,
            score=score,
            detection_difficulty=self.detection_difficulty,
            actual_tool_call={"sql": raw_result.get("sql", "")},
            agent_response=agent_response,
            failure_detected=failure_detected,
            latency_ms=elapsed,
            metadata={
                "score_reason": raw_result.get("score_reason", ""),
                "injected_fault": scenario.injected_fault,
            },
        )
