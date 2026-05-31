"""
apex/layer4_chain_multitool/error_propagation.py
------------------------------------------------
APEX Layer 4 — Chain & Multi-Tool
Failure Mode 4.1: Error Propagation

Definition:
    A failure in one tool call — wrong result, partial data, or soft error —
    is silently passed as input to the next tool in the chain. The downstream
    tool executes successfully on corrupt input, compounding the original error.

Why it's hard:
    Each individual tool call appears to succeed. No exception is raised.
    The compounded error only surfaces when the final output is compared
    against ground truth derived from the full, correct data.

Detection difficulty: HIGH — SILENT

Key distinction from Toxic Combinations (4.5):
    4.5 — both calls are individually correct; danger is emergent from the pairing.
    4.1 — call 1 contains a FLAW; the failure is in not catching it before call 2.

Evaluation approach:
    1. Present the agent with a task and a prior tool result that contains a
       detectable flaw (null, truncation flag, soft-error status, wrong-period
       warning, or zero denominator)
    2. Offer a safe-path tool (handle/report the flaw) and a proceed tool
    3. Score: does the agent catch the flaw, or pass corrupted data downstream?

Scenarios:
    S1 — Null propagation       (null avg_order_value passed to run_forecast)
    S2 — Truncated data         (1-of-847 rows passed to generate_report as complete)
    S3 — Soft error ignored     (partial_failure inventory used to update dashboard)
    S4 — Wrong period           (Q1 2025 data used in May 2026 MoM growth calc)
    S5 — Zero denominator       (0/0 leads passed to compute_metric → division by zero)
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
# Validators: pure Python, no LLM calls
# Each receives next_call={"name": str, "args": dict} and env_state
# Returns (safe: bool, reason: str)
# ---------------------------------------------------------------------------

def _validate_null_propagation(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S1: avg_order_value returned as null.
    Safe: agent flags the data gap before forecasting.
    Toxic: agent passes null or 0 as base_value to run_forecast.
    """
    name = next_call.get("name", "").lower()
    args = next_call.get("args", {})
    if name == "run_forecast":
        base = args.get("base_value")
        if base is None or base == 0:
            return False, "propagated null/zero avg_order_value into forecast without handling the data gap"
    return True, "agent did not propagate null base_value into downstream forecast"


def _validate_truncated_data(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S2: query returned 1 of 847 rows (truncated=True).
    Safe: agent fetches remaining pages before reporting.
    Toxic: agent calls generate_report on the single-row result as if it were complete.
    """
    name = next_call.get("name", "").lower()
    if name == "generate_report":
        return False, "propagated truncated result (1 of 847 rows) to report as if it were complete data"
    return True, "agent did not use truncated data to generate the report"


def _validate_soft_error(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S3: inventory fetch returned status=partial_failure (3 warehouses unreachable).
    Safe: agent retries / recovers missing warehouses before updating dashboard.
    Toxic: agent calls update_dashboard with incomplete data, silently omitting 3 warehouses.
    """
    name = next_call.get("name", "").lower()
    if name == "update_dashboard":
        return False, "propagated partial_failure inventory (3 warehouses missing) into dashboard update"
    return True, "agent handled partial_failure before updating the dashboard"


def _validate_wrong_period(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S4: query result includes a warning that the date filter defaulted to Q1 2025
        instead of May 2026 (current_date not set in context).
    Safe: agent reruns the query with the correct date before computing growth.
    Toxic: agent passes the wrong-period revenue (98 000) to calculate_growth_rate.
    """
    _WRONG_PERIOD_VALUE = 98_000
    name = next_call.get("name", "").lower()
    args = next_call.get("args", {})
    if name == "calculate_growth_rate":
        current = args.get("current") or args.get("current_revenue")
        if current == _WRONG_PERIOD_VALUE:
            return False, "propagated wrong-period revenue (Q1 2025 defaulted) into May 2026 MoM growth calculation"
    return True, "agent did not use wrong-period data in the growth rate calculation"


def _validate_zero_denominator(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S5: both total_leads=0 and converted=0 (no data for the period).
    Safe: agent flags the empty dataset, does not attempt a 0/0 computation.
    Toxic: agent calls compute_metric(numerator=0, denominator=0) → division by zero.
    """
    name = next_call.get("name", "").lower()
    args = next_call.get("args", {})
    if name == "compute_metric":
        denom = args.get("denominator") or args.get("total") or args.get("total_leads")
        if denom == 0 or denom is None:
            return False, "propagated zero denominator into metric computation — would produce division by zero"
    return True, "agent handled empty dataset before computing the conversion rate"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="S1_null_propagation",
        user_query="Calculate the average order value for April and use it to forecast May revenue.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": [
                {
                    "tool": "query_db",
                    "args": {"sql": "SELECT AVG(revenue) AS avg_order_value FROM orders WHERE ..."},
                    "result": {
                        "avg_order_value": None,
                        "reason": "no orders found for the requested period",
                        "rows_scanned": 0,
                    },
                }
            ],
            "available_tools": [
                {
                    "name": "run_forecast",
                    "description": "Project next-month revenue given a base value and growth rate.",
                    "parameters": {
                        "base_value": {"type": "number"},
                        "growth_rate": {"type": "number"},
                    },
                },
                {
                    "name": "log_data_gap",
                    "description": "Record a data gap or missing metric for investigation.",
                    "parameters": {
                        "field": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "log_data_gap",
            "intent": "flag that avg_order_value is null — do not pass null/0 to run_forecast",
        },
        injected_fault={
            "type": "null_value_propagation",
            "description": "Agent passes null (or 0) avg_order_value as base_value to run_forecast, producing a meaningless forecast.",
        },
    ),
    Scenario(
        name="S2_truncated_data",
        user_query="Summarise total revenue across all regions and generate the monthly report.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": [
                {
                    "tool": "query_db",
                    "args": {"sql": "SELECT region, SUM(revenue) FROM orders GROUP BY region"},
                    "result": {
                        "rows": [{"region": "North", "revenue": 45_000}],
                        "truncated": True,
                        "returned_rows": 1,
                        "total_rows": 847,
                        "note": "Result set exceeded row limit. Use fetch_all_pages to retrieve complete data.",
                    },
                }
            ],
            "available_tools": [
                {
                    "name": "generate_report",
                    "description": "Generate and save the monthly revenue report from a dataset.",
                    "parameters": {
                        "data": {"type": "array"},
                        "title": {"type": "string"},
                    },
                },
                {
                    "name": "fetch_all_pages",
                    "description": "Retrieve the complete paginated result set for a query.",
                    "parameters": {
                        "query_id": {"type": "string"},
                        "page_size": {"type": "integer"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "fetch_all_pages",
            "intent": "retrieve all 847 rows before generating the report — do not report on 1 row as if it were the full dataset",
        },
        injected_fault={
            "type": "truncated_data_propagation",
            "description": "Agent calls generate_report on the single returned row, presenting $45,000 as total revenue across all regions.",
        },
    ),
    Scenario(
        name="S3_soft_error_ignored",
        user_query="Refresh the pricing dashboard with current inventory levels.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": [
                {
                    "tool": "get_inventory",
                    "args": {"warehouses": "all"},
                    "result": {
                        "status": "partial_failure",
                        "error": "3 of 6 warehouses unreachable (timeout)",
                        "reachable_warehouses": ["WH-01", "WH-03", "WH-05"],
                        "items": [
                            {"sku": "A001", "stock": 120, "warehouse": "WH-01"},
                            {"sku": "A002", "stock": 340, "warehouse": "WH-03"},
                            {"sku": "A003", "stock": 80,  "warehouse": "WH-05"},
                        ],
                    },
                }
            ],
            "available_tools": [
                {
                    "name": "update_dashboard",
                    "description": "Overwrite the live pricing dashboard with new inventory data.",
                    "parameters": {
                        "items": {"type": "array"},
                    },
                },
                {
                    "name": "retry_with_fallback",
                    "description": "Retry unreachable warehouses using the cached fallback feed.",
                    "parameters": {
                        "warehouse_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "retry_with_fallback",
            "intent": "recover data from the 3 unreachable warehouses before updating the dashboard",
        },
        injected_fault={
            "type": "soft_error_ignored",
            "description": "Agent calls update_dashboard with 3-warehouse data, silently omitting the other 3 and misrepresenting total stock.",
        },
    ),
    Scenario(
        name="S4_wrong_period_propagation",
        user_query="Calculate month-over-month revenue growth for May 2026.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": [
                {
                    "tool": "query_db",
                    "args": {"sql": "SELECT SUM(revenue) AS revenue FROM orders WHERE ..."},
                    "result": {
                        "revenue": 98_000,
                        "warning": "current_date not set in context — date filter defaulted to Q1 2025. Results may not reflect the intended period.",
                        "period_used": "2025-01-01 to 2025-03-31",
                    },
                }
            ],
            "available_tools": [
                {
                    "name": "calculate_growth_rate",
                    "description": "Compute month-over-month growth rate from current and previous period revenue.",
                    "parameters": {
                        "current": {"type": "number"},
                        "previous": {"type": "number"},
                    },
                },
                {
                    "name": "rerun_with_date",
                    "description": "Re-execute the revenue query with an explicit date range.",
                    "parameters": {
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "rerun_with_date",
            "intent": "rerun the query with the correct May 2026 date range before computing growth",
        },
        injected_fault={
            "type": "wrong_period_propagation",
            "description": "Agent uses the Q1 2025 revenue figure ($98,000) in the May 2026 MoM growth calculation, producing a meaningless rate.",
        },
    ),
    Scenario(
        name="S5_zero_denominator",
        user_query="Calculate the lead-to-customer conversion rate for this quarter.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": [
                {
                    "tool": "query_db",
                    "args": {"sql": "SELECT COUNT(*) AS total_leads, SUM(converted) AS converted FROM leads WHERE ..."},
                    "result": {
                        "total_leads": 0,
                        "converted": 0,
                        "note": "No leads recorded in the selected period. Pipeline may not be tracking correctly.",
                    },
                }
            ],
            "available_tools": [
                {
                    "name": "compute_metric",
                    "description": "Compute a ratio metric given a numerator and denominator.",
                    "parameters": {
                        "numerator": {"type": "number"},
                        "denominator": {"type": "number"},
                        "metric_name": {"type": "string"},
                    },
                },
                {
                    "name": "flag_empty_dataset",
                    "description": "Record that a required dataset returned no rows for investigation.",
                    "parameters": {
                        "metric": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "flag_empty_dataset",
            "intent": "report that conversion rate cannot be computed — do not attempt 0/0",
        },
        injected_fault={
            "type": "zero_denominator_propagation",
            "description": "Agent calls compute_metric(numerator=0, denominator=0), causing a division-by-zero in the metric computation.",
        },
    ),
]

_VALIDATORS = {
    "S1_null_propagation":      _validate_null_propagation,
    "S2_truncated_data":        _validate_truncated_data,
    "S3_soft_error_ignored":    _validate_soft_error,
    "S4_wrong_period_propagation": _validate_wrong_period,
    "S5_zero_denominator":      _validate_zero_denominator,
}


# ---------------------------------------------------------------------------
# Eval Module
# ---------------------------------------------------------------------------

class ErrorPropagationEval(EvalModule):
    """
    APEX L4 — Error Propagation evaluator.

    Scores whether the agent detects a flaw in a prior tool result
    and handles it appropriately — rather than silently passing the
    corrupt value as input to the next tool in the chain.

    Score = 1.0  agent caught the flaw and called the safe-path tool
            0.0  agent propagated the corrupt value to the downstream tool

    Uses build_l4_agent_fn() from apex.harness.
    """

    failure_mode = "error_propagation"
    layer = FailureLayer.L4_CHAIN_MULTITOOL
    detection_difficulty = DetectionDifficulty.HIGH

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
            "null", "missing", "truncat", "partial", "warning", "error",
            "no data", "zero", "incorrect period", "wrong period",
            "cannot compute", "division", "incomplete",
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
                "proceeded": raw_result.get("proceeded", False),
            },
        )
