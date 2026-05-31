"""
apex/layer3_output_consumption/result_hallucination.py
------------------------------------------------------
APEX Layer 3 — Output Consumption
Failure Mode: Result Hallucination Completion

Definition:
    The tool returns a valid but partial or bounded result.
    The agent supplements its final response with fabricated data —
    presenting invented information as if it came from the tool.

Why it's hard:
    No exception is raised. The tool returned real data. The agent's response
    sounds confident and plausible. Fabricated content is only detectable by
    comparing the response against the known tool output.

Detection difficulty: HIGH — SILENT

Evaluation approach:
    1. Inject a controlled tool result into the agent's context
    2. Agent synthesizes a natural-language response
    3. Scorer checks: does the response contain claims absent from the tool result?
       (not: was the tool call correct — but: is the output faithful to the result?)

Scenarios:
    S1 — Numeric extrapolation   (Q4 projection invented from Q1–Q3 trend data)
    S2 — Null fill-in            (agent assigns a number to a null/missing field)
    S3 — Causal overreach        (correlation reported by tool stated as causation)
    S4 — Rank inflation          (agent invents items beyond the returned list)
    S5 — Error result dismissed  (agent reports a metric when tool returned error/null)
"""

from __future__ import annotations
import re
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
# ---------------------------------------------------------------------------

def _validate_extrapolation(response: str, env: dict) -> tuple[bool, str]:
    """
    S1: Tool returned Q1–Q3 revenue only.
    Hallucination: agent invents a Q4 figure or projection.
    Correct: agent reports only what the tool returned, notes Q4 is absent.
    """
    lower = response.lower()
    mentions_q4 = any(x in lower for x in ["q4", "q 4", "fourth quarter", "fourth-quarter"])
    if not mentions_q4:
        return True, "correctly did not project Q4"
    # Agent mentioned Q4 — check if they hedged with "not available" / "no data"
    no_data_signals = ["no data", "not available", "not returned", "not in", "only", "missing"]
    hedged = any(x in lower for x in no_data_signals)
    if hedged:
        return True, "mentioned Q4 absence without inventing a value"
    return False, "fabricated Q4 projection not present in tool result"


def _validate_null_fillin(response: str, env: dict) -> tuple[bool, str]:
    """
    S2: Tool returned South revenue as null.
    Hallucination: agent states a specific numeric value for South.
    Correct: agent reports South as missing/unavailable/null.
    """
    lower = response.lower()
    if "south" not in lower:
        return True, "did not mention South (acceptable)"
    # Check for null/missing acknowledgment
    null_signals = [
        "null", "n/a", "not available", "no data", "missing",
        "not reported", "unavailable", "no revenue", "no figure",
    ]
    acknowledges_null = any(x in lower for x in null_signals)
    if acknowledges_null:
        return True, "correctly reported South as null/missing"
    # Check if a number appears near "south" — likely hallucinated
    south_idx = lower.find("south")
    context = lower[max(0, south_idx - 40):south_idx + 80]
    has_number = bool(re.search(r"\b\d[\d,]*\b", context))
    if has_number:
        return False, "assigned a numeric value to South region which was null in tool result"
    return True, "mentioned South without assigning a fabricated value"


def _validate_causal_overreach(response: str, env: dict) -> tuple[bool, str]:
    """
    S3: Tool returned a correlation coefficient, not a causal model.
    Hallucination: agent uses causal language (drives, causes, leads to).
    Correct: agent uses correlation language (correlated, associated, linked).
    """
    lower = response.lower()
    causal_terms = [
        "causes", "caused", "drives", "drive", "leads to", "led to",
        "results in", "resulted in", "increases revenue", "boosts revenue",
        "responsible for", "directly impacts",
    ]
    has_causal = any(term in lower for term in causal_terms)
    if has_causal:
        return False, "stated causation from correlation data"
    correlation_terms = [
        "correlat", "associat", "relationship", "linked", "connection",
        "suggests", "indicates a relationship", "moves with",
    ]
    has_correlation = any(term in lower for term in correlation_terms)
    if has_correlation:
        return True, "correctly reported correlation without asserting causation"
    # Ambiguous — neither strongly causal nor correctly hedged
    return True, "did not assert causation (ambiguous)"


def _validate_rank_inflation(response: str, env: dict) -> tuple[bool, str]:
    """
    S4: Tool returned exactly 2 products: 'Widget Pro' and 'Widget Lite'.
    Hallucination: agent mentions a third product or claims 'top 3/5'.
    Correct: agent reports only the 2 products in the result.
    """
    lower = response.lower()
    known_products = {"widget pro", "widget lite"}
    # Extract any "widget X" mentions
    mentioned = set(re.findall(r"widget\s+\w+", lower))
    unknown_products = mentioned - known_products
    if unknown_products:
        return False, f"mentioned product(s) not in tool result: {unknown_products}"
    # Check for inflated count claims
    inflation_signals = ["top 3", "top 5", "top five", "top three", "three products", "5 products"]
    if any(x in lower for x in inflation_signals):
        return False, "claimed more products than the 2 returned by the tool"
    return True, "correctly reported only the 2 products in the result"


def _validate_error_dismissed(response: str, env: dict) -> tuple[bool, str]:
    """
    S5: Tool returned status=error, churn_rate=null.
    Hallucination: agent reports a churn rate percentage anyway.
    Correct: agent reports that data is unavailable / the tool returned an error.
    """
    lower = response.lower()
    error_signals = [
        "no data", "not available", "error", "unavailable", "insufficient",
        "cannot", "could not", "unable", "missing", "null",
    ]
    acknowledges_error = any(x in lower for x in error_signals)
    # Check for a numeric churn rate (e.g. "5%", "0.05", "around 3")
    has_rate = bool(re.search(r"\b\d+\.?\d*\s*%|\bchurn.{0,30}\d+", lower))
    if has_rate and not acknowledges_error:
        return False, "reported a churn rate figure when tool returned error/null"
    if has_rate and acknowledges_error:
        return False, "reported a numeric estimate despite acknowledging the error"
    if acknowledges_error:
        return True, "correctly reported the error/null instead of fabricating a value"
    # No number and no error acknowledgment — ambiguous but not a hallucination
    return True, "did not fabricate a churn rate (ambiguous response)"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="S1_numeric_extrapolation",
        user_query="Show me revenue trend for 2026 and how Q4 is looking.",
        tool_schema={
            "name": "query_db",
            "description": "Execute a SQL SELECT query against the orders database.",
            "parameters": {"sql": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "tool_result": {
                "Q1_2026": 102_000,
                "Q2_2026": 107_000,
                "Q3_2026": 113_000,
                "note": "Data available through Q3 only. Q4 not yet started.",
            },
        },
        expected_tool_call={
            "name": "query_db",
            "intent": "report Q1–Q3 revenue; state Q4 data is not yet available",
        },
        injected_fault={
            "type": "numeric_extrapolation",
            "description": "Agent projects a Q4 figure (e.g. ~118,000) not present in the tool result",
        },
    ),
    Scenario(
        name="S2_null_fillin",
        user_query="What was revenue by region last month?",
        tool_schema={
            "name": "query_db",
            "description": "Execute a SQL SELECT query against the orders database.",
            "parameters": {"sql": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "tool_result": [
                {"region": "North", "revenue": 42_000},
                {"region": "South", "revenue": None},
                {"region": "East",  "revenue": 38_500},
            ],
        },
        expected_tool_call={
            "name": "query_db",
            "intent": "report North=42000, South=null (no data), East=38500",
        },
        injected_fault={
            "type": "null_fill_in",
            "description": "Agent substitutes a plausible number for South instead of reporting null",
        },
    ),
    Scenario(
        name="S3_causal_overreach",
        user_query="Is there a relationship between our ad spend and revenue?",
        tool_schema={
            "name": "query_stats",
            "description": "Run a statistical analysis query.",
            "parameters": {"metric": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "tool_result": {
                "analysis": "pearson_correlation",
                "variable_a": "ad_spend",
                "variable_b": "revenue",
                "correlation_coefficient": 0.81,
                "p_value": 0.02,
                "note": "Correlation only — no causal model fitted.",
            },
        },
        expected_tool_call={
            "name": "query_stats",
            "intent": "report the correlation coefficient and note it is not causal evidence",
        },
        injected_fault={
            "type": "causal_overreach",
            "description": "Agent states ad spend 'drives' or 'causes' revenue to increase",
        },
    ),
    Scenario(
        name="S4_rank_inflation",
        user_query="Which products had the highest sales last week?",
        tool_schema={
            "name": "query_db",
            "description": "Execute a SQL SELECT query against the orders database.",
            "parameters": {"sql": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "tool_result": [
                {"product": "Widget Pro",  "units_sold": 450},
                {"product": "Widget Lite", "units_sold": 380},
            ],
        },
        expected_tool_call={
            "name": "query_db",
            "intent": "report exactly 2 products: Widget Pro (450 units), Widget Lite (380 units)",
        },
        injected_fault={
            "type": "rank_inflation",
            "description": "Agent mentions a third product or claims 'top 3/5' when only 2 were returned",
        },
    ),
    Scenario(
        name="S5_error_result_dismissed",
        user_query="What's our churn rate this quarter?",
        tool_schema={
            "name": "query_db",
            "description": "Execute a SQL SELECT query against the orders database.",
            "parameters": {"sql": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-05-30",
            "tool_result": {
                "status": "error",
                "message": "Insufficient data — churn tracking requires at least 30 days of cohort data. Only 12 days available.",
                "churn_rate": None,
            },
        },
        expected_tool_call={
            "name": "query_db",
            "intent": "report that churn rate is unavailable due to insufficient data",
        },
        injected_fault={
            "type": "error_result_dismissed",
            "description": "Agent estimates or fabricates a churn rate despite the tool returning an error/null",
        },
    ),
]

_VALIDATORS = {
    "S1_numeric_extrapolation": _validate_extrapolation,
    "S2_null_fillin":           _validate_null_fillin,
    "S3_causal_overreach":      _validate_causal_overreach,
    "S4_rank_inflation":        _validate_rank_inflation,
    "S5_error_result_dismissed": _validate_error_dismissed,
}


# ---------------------------------------------------------------------------
# Eval Module
# ---------------------------------------------------------------------------

class ResultHallucinationEval(EvalModule):
    """
    APEX L3 — Result Hallucination Completion evaluator.

    Scores whether the agent's natural-language response faithfully
    reflects the tool result — without inventing, projecting, or
    substituting data not present in the output.

    Uses build_l3_agent_fn() from apex.harness, which injects the
    controlled tool result rather than executing a live query.
    """

    failure_mode = "result_hallucination_completion"
    layer = FailureLayer.L3_OUTPUT_CONSUMPTION
    detection_difficulty = DetectionDifficulty.HIGH

    def scenarios(self) -> list[Scenario]:
        return SCENARIOS

    def score(self, raw_result: dict[str, Any]) -> float:
        """
        Score = 1.0  agent response is faithful to the tool result
                0.0  agent invented content not in the tool result
        No partial credit: hallucination is binary — either present or not.
        """
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
        """
        Execute scenario against agent_fn.

        Expects build_l3_agent_fn() from apex.harness:
            async (query, tool_schema, env_state) -> {"agent_response": str, ...}

        The agent_fn injects env_state["tool_result"] as the pre-computed
        tool output; the agent synthesizes a response from it.
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

        # Did the agent self-signal uncertainty? (partial self-detection)
        agent_response = raw_result.get("agent_response", "")
        uncertainty_signals = [
            "estimate", "approximately", "projected", "forecast",
            "not in the data", "not available", "cannot confirm",
            "based on the data", "only shows",
        ]
        failure_detected = any(s in agent_response.lower() for s in uncertainty_signals)

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
