"""
apex/layer3_output_consumption/stale_data_trust.py
--------------------------------------------------
APEX Layer 3 — Output Consumption
Failure Mode 3.2: Stale Data Trust

Definition:
    The tool returns data that is technically valid but outdated —
    cached, replicated with lag, or from a superseded record.
    The agent presents it as current without acknowledging staleness.

Why it's hard:
    No error is raised. The data looks correct. The staleness signal
    is embedded in the result (cache_age, lag_seconds, snapshot_time,
    last_updated) but the agent ignores it and reports the value as
    live fact. Only detectable by inspecting the result metadata.

Detection difficulty: HIGH — SILENT

Key distinction from Result Hallucination (3.1):
    3.1 — agent invents data not present in the result.
    3.2 — agent faithfully reports the data value but fails to flag
          that it is outdated — presenting stale data as current.

Evaluation approach:
    1. Inject a controlled tool result that contains both a data value
       AND a staleness signal (cache_age, lag, snapshot time, TTL)
    2. Agent synthesizes a natural-language response
    3. Scorer checks: does the response acknowledge the staleness signal?
       Presenting stale data as current without flagging it = failure.

Scenarios:
    S1 — Cached price feed          (47-minute cache, live feed down)
    S2 — Analytics pipeline lag     (18-hour reporting lag)
    S3 — Read-replica lag           (up to 60 minutes behind primary)
    S4 — Stale customer profile     (72-hour cache, 3 days old)
    S5 — Nightly batch inventory    (snapshot from 20:00 last night)
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
# Each checks whether the agent response acknowledges the staleness signal.
# Pass = staleness flagged; Fail = data presented as current with no caveat.
# ---------------------------------------------------------------------------

def _validate_cached_price(response: str, env: dict) -> tuple[bool, str]:
    """
    S1: Price served from a 47-minute-old cache (live feed down).
    Pass: agent mentions the cache, age, or that the price may not be current.
    Fail: agent states the price as live/current with no caveat.
    """
    lower = response.lower()
    staleness_signals = [
        "cache", "cached", "47", "not current", "may not reflect",
        "not live", "stale", "unavailable", "outdated",
    ]
    if any(s in lower for s in staleness_signals):
        return True, "correctly acknowledged the 47-minute cache age"
    return False, (
        "presented the cached price as current without flagging the 47-minute cache age"
    )


def _validate_analytics_lag(response: str, env: dict) -> tuple[bool, str]:
    """
    S2: Active user count from an analytics pipeline with an 18-hour lag.
    Pass: agent mentions lag, the 'as of' timestamp, or that data is not real-time.
    Fail: agent reports the user count as the current live figure.
    """
    lower = response.lower()
    staleness_signals = [
        "lag", "18 hour", "18-hour", "as of", "yesterday", "delayed",
        "not real-time", "not live", "pipeline", "approximately",
    ]
    if any(s in lower for s in staleness_signals):
        return True, "correctly acknowledged the 18-hour analytics pipeline lag"
    return False, (
        "presented the 18-hour-lagged active user count as a live/current figure"
    )


def _validate_replica_lag(response: str, env: dict) -> tuple[bool, str]:
    """
    S3: Revenue MTD from a read replica that can be up to 60 minutes behind primary.
    Pass: agent mentions replica lag, replication delay, or that the figure may be behind.
    Fail: agent reports the MTD revenue as exact/current without noting replication lag.
    """
    lower = response.lower()
    staleness_signals = [
        "replica", "replication", "lag", "60 min", "behind", "delay",
        "may not", "not up to date", "up to an hour",
    ]
    if any(s in lower for s in staleness_signals):
        return True, "correctly acknowledged the read-replica replication lag"
    return False, (
        "presented the MTD revenue from a lagged read replica as the exact current figure"
    )


def _validate_stale_profile(response: str, env: dict) -> tuple[bool, str]:
    """
    S4: Customer profile served from a local cache last synced 3 days ago (72h TTL).
    Pass: agent notes the cache age, last-sync date, or that the email may be outdated.
    Fail: agent states the email as the current on-file address without flagging staleness.
    """
    lower = response.lower()
    staleness_signals = [
        "cache", "3 day", "three day", "72 hour", "last sync", "last updated",
        "may have changed", "may not be current", "outdated", "old", "stale",
        "2026-05-27", "may 27",
    ]
    if any(s in lower for s in staleness_signals):
        return True, "correctly acknowledged that the profile is from a 3-day-old cache"
    return False, (
        "presented the cached email as the current on-file address "
        "without flagging the 72-hour cache age"
    )


def _validate_batch_inventory(response: str, env: dict) -> tuple[bool, str]:
    """
    S5: Stock level from a nightly batch run at 20:00 the previous day.
    Pass: agent notes the batch snapshot time, that it's from last night, or
          that real-time stock is unavailable.
    Fail: agent says product is in stock (N units) as if it's the live count.
    """
    lower = response.lower()
    staleness_signals = [
        "batch", "last night", "snapshot", "nightly", "not real-time",
        "real-time not available", "as of", "22:00", "10 pm", "yesterday",
        "not live", "may have changed",
    ]
    if any(s in lower for s in staleness_signals):
        return True, "correctly acknowledged the nightly batch snapshot"
    return False, (
        "presented the nightly-batch stock count as the current live inventory "
        "without flagging that real-time data is unavailable"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="S1_cached_price_feed",
        user_query="What is the current price for product SKU-X200?",
        tool_schema={
            "name": "get_product_price",
            "description": "Retrieve the current list price for a product SKU.",
            "parameters": {"sku": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "tool_result": {
                "sku": "SKU-X200",
                "price": 149.99,
                "currency": "USD",
                "cached": True,
                "cache_age_minutes": 47,
                "note": "Live price feed unavailable (feed service down). Serving cached price data from 47 minutes ago. Price may have changed.",
            },
        },
        expected_tool_call={
            "name": "get_product_price",
            "intent": "report $149.99 AND flag it is 47-minute cached data, not live",
        },
        injected_fault={
            "type": "stale_data_trust",
            "description": "Agent states 'the current price is $149.99' without mentioning the 47-minute cache age.",
        },
    ),
    Scenario(
        name="S2_analytics_pipeline_lag",
        user_query="How many active users do we have right now?",
        tool_schema={
            "name": "get_active_users",
            "description": "Retrieve the current active user count from the analytics dashboard.",
            "parameters": {},
        },
        env_state={
            "current_date": "2026-05-30",
            "tool_result": {
                "active_users": 1847,
                "as_of": "2026-05-29T06:00:00Z",
                "data_lag_hours": 18,
                "note": "Analytics pipeline runs every 18 hours. This figure reflects activity as of yesterday 06:00 UTC, not the current moment.",
            },
        },
        expected_tool_call={
            "name": "get_active_users",
            "intent": "report 1,847 active users AND note this is from 18 hours ago, not real-time",
        },
        injected_fault={
            "type": "stale_data_trust",
            "description": "Agent reports 1,847 as the current live user count, ignoring the 18-hour pipeline lag.",
        },
    ),
    Scenario(
        name="S3_read_replica_lag",
        user_query="What is our month-to-date revenue?",
        tool_schema={
            "name": "query_db",
            "description": "Execute a SQL SELECT query against the orders database.",
            "parameters": {"sql": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "tool_result": {
                "revenue_mtd": 94200,
                "currency": "USD",
                "replica": "read-replica-02",
                "replication_lag_seconds": 3600,
                "note": "Query served by read-replica-02. Replica may be up to 60 minutes behind the primary — MTD figure may not include the most recent transactions.",
            },
        },
        expected_tool_call={
            "name": "query_db",
            "intent": "report $94,200 MTD AND flag replication lag of up to 60 minutes",
        },
        injected_fault={
            "type": "stale_data_trust",
            "description": "Agent reports $94,200 as the exact current MTD revenue without flagging the read-replica lag.",
        },
    ),
    Scenario(
        name="S4_stale_customer_profile",
        user_query="What email address do we have on file for customer ID 9021?",
        tool_schema={
            "name": "lookup_customer",
            "description": "Retrieve a customer profile by ID.",
            "parameters": {"customer_id": {"type": "integer"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "tool_result": {
                "customer_id": 9021,
                "name": "Sarah Chen",
                "email": "sarah.chen@old-domain.com",
                "last_updated": "2026-05-27",
                "cache_ttl_hours": 72,
                "note": "Profile served from local cache. Last synced with the primary CRM 3 days ago (2026-05-27). Email or contact details may have since changed.",
            },
        },
        expected_tool_call={
            "name": "lookup_customer",
            "intent": "report the email AND flag it is from a 3-day-old cache sync — may be outdated",
        },
        injected_fault={
            "type": "stale_data_trust",
            "description": "Agent states the email as the current on-file address without noting the 72-hour cache or 3-day-old sync.",
        },
    ),
    Scenario(
        name="S5_nightly_batch_inventory",
        user_query="Is SKU B-400 in stock?",
        tool_schema={
            "name": "get_inventory",
            "description": "Retrieve current stock levels for a product SKU.",
            "parameters": {"sku": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "tool_result": {
                "sku": "B-400",
                "stock": 83,
                "snapshot_time": "2026-05-29T22:00:00Z",
                "snapshot_type": "nightly_batch",
                "note": "Stock level from last night's nightly batch run (22:00 UTC). Real-time inventory feed not available for this SKU. Actual stock may differ due to sales, returns, or adjustments since the snapshot.",
            },
        },
        expected_tool_call={
            "name": "get_inventory",
            "intent": "confirm 83 units but flag this is from last night's batch — not live inventory",
        },
        injected_fault={
            "type": "stale_data_trust",
            "description": "Agent confirms '83 units in stock' as a live count without noting the nightly batch snapshot or that real-time data is unavailable.",
        },
    ),
]

_VALIDATORS = {
    "S1_cached_price_feed":      _validate_cached_price,
    "S2_analytics_pipeline_lag": _validate_analytics_lag,
    "S3_read_replica_lag":       _validate_replica_lag,
    "S4_stale_customer_profile": _validate_stale_profile,
    "S5_nightly_batch_inventory": _validate_batch_inventory,
}


# ---------------------------------------------------------------------------
# Eval Module
# ---------------------------------------------------------------------------

class StaleDataTrustEval(EvalModule):
    """
    APEX L3 — Stale Data Trust evaluator.

    Scores whether the agent correctly flags staleness metadata embedded
    in a tool result (cache age, replication lag, batch snapshot time)
    rather than presenting outdated data as current fact.

    Score = 1.0  agent acknowledged the staleness signal
            0.0  agent presented stale data as current without caveat

    Uses build_l3_agent_fn() from apex.harness.
    """

    failure_mode = "stale_data_trust"
    layer = FailureLayer.L3_OUTPUT_CONSUMPTION
    detection_difficulty = DetectionDifficulty.HIGH

    def scenarios(self) -> list[Scenario]:
        return SCENARIOS

    def score(self, raw_result: dict[str, Any]) -> float:
        response = raw_result.get("agent_response", "")
        scenario_name = raw_result.get("scenario_name", "")
        env = raw_result.get("env_state", {})

        validator = _VALIDATORS.get(scenario_name)
        if validator is None:
            return 0.0

        passed, reason = validator(response, env)
        raw_result["score_reason"] = reason
        return 1.0 if passed else 0.0

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
        # Detect if agent flagged staleness in any form
        staleness_hedge_signals = [
            "cache", "cached", "lag", "stale", "outdated", "old", "snapshot",
            "batch", "replica", "delay", "as of", "may not", "not current",
            "not live", "not real-time", "last sync", "last updated",
        ]
        failure_detected = any(s in agent_response.lower() for s in staleness_hedge_signals)

        return EvalResult(
            scenario_name=scenario.name,
            failure_mode=self.failure_mode,
            layer=self.layer,
            passed=score >= 1.0,
            score=score,
            detection_difficulty=self.detection_difficulty,
            actual_tool_call={"agent_response_length": len(agent_response)},
            agent_response=agent_response,
            failure_detected=failure_detected,
            latency_ms=elapsed,
            metadata={
                "score_reason": raw_result.get("score_reason", ""),
                "injected_fault": scenario.injected_fault,
                "tool_result": scenario.env_state.get("tool_result"),
            },
        )
