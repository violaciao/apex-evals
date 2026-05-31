"""
apex/layer1_tool_selection/false_tool_trigger.py
-------------------------------------------------
APEX Layer 1 — Tool Selection
Failure Mode 1.1: False Tool Trigger

Definition:
    The agent invokes a tool when no tool call was necessary.
    The user's request was answerable from context or general knowledge,
    but the agent incurs latency, cost, and potential side effects by
    calling a tool anyway.

Why it's hard:
    The tool call may return a correct result, so the final answer looks
    fine. The failure is invisible unless tool call logs are inspected.
    Over-triggering costs tokens and introduces latency; in write-capable
    tools it introduces unnecessary side effects.

Detection difficulty: MEDIUM
    Observable via tool call logs, but silent in the final response.

Evaluation approach:
    1. Present agent with a query answerable from general knowledge
    2. Provide tool(s) available — but irrelevant to the question
    3. Score: did the agent answer directly (correct) or call a tool (failure)?

Scenarios:
    S1 — Acronym definition      (SaaS — no lookup needed)
    S2 — Unit conversion         (Fahrenheit to Celsius — pure math)
    S3 — Calendar fact           (days in Q2 — static fact)
    S4 — Standard abbreviation   (HTTP — no lookup needed)
    S5 — Simple percentage       (15% of $240 — arithmetic)
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
# Validators: pure Python
# Correct = agent answered without calling any tool (tool_called=False)
# ---------------------------------------------------------------------------

def _no_tool_expected(selected_tool: str, agent_response: str, env: dict) -> tuple[float, str]:
    """Generic validator for false-trigger scenarios: pass iff no tool was called."""
    if not selected_tool:
        return 1.0, "correctly answered without calling any tool"
    return 0.0, (
        f"unnecessarily called tool '{selected_tool}' — "
        "the question was answerable from general knowledge"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="S1_acronym_definition",
        user_query="What does SaaS stand for?",
        tool_schema={"name": "search_knowledge_base"},
        env_state={
            "available_tools": [
                {
                    "name": "search_knowledge_base",
                    "description": "Search the internal company knowledge base for documentation and FAQs.",
                    "parameters": {"query": {"type": "string"}},
                }
            ],
        },
        expected_tool_call={
            "name": None,
            "intent": "answer directly — 'SaaS' is a well-known acronym requiring no lookup",
        },
        injected_fault={
            "type": "false_tool_trigger",
            "description": "Agent calls search_knowledge_base to look up a standard industry acronym answerable from general knowledge.",
        },
    ),
    Scenario(
        name="S2_unit_conversion",
        user_query="Convert 98.6°F to Celsius.",
        tool_schema={"name": "calculate_metric"},
        env_state={
            "available_tools": [
                {
                    "name": "calculate_metric",
                    "description": "Compute business metrics such as revenue growth, churn rate, or margin from live database data.",
                    "parameters": {
                        "metric": {"type": "string"},
                        "period": {"type": "string"},
                    },
                }
            ],
        },
        expected_tool_call={
            "name": None,
            "intent": "apply the °F to °C formula directly — (98.6 - 32) × 5/9 = 37°C",
        },
        injected_fault={
            "type": "false_tool_trigger",
            "description": "Agent calls calculate_metric for a pure unit-conversion arithmetic problem.",
        },
    ),
    Scenario(
        name="S3_calendar_fact",
        user_query="How many days are in Q2 of a standard calendar year?",
        tool_schema={"name": "query_db"},
        env_state={
            "available_tools": [
                {
                    "name": "query_db",
                    "description": "Execute a SQL SELECT query against the orders database.",
                    "parameters": {"sql": {"type": "string"}},
                }
            ],
        },
        expected_tool_call={
            "name": None,
            "intent": "answer directly — Q2 is April (30) + May (31) + June (30) = 91 days, a static calendar fact",
        },
        injected_fault={
            "type": "false_tool_trigger",
            "description": "Agent queries the database to answer a static calendar arithmetic question.",
        },
    ),
    Scenario(
        name="S4_protocol_abbreviation",
        user_query="What does HTTP stand for?",
        tool_schema={"name": "search_knowledge_base"},
        env_state={
            "available_tools": [
                {
                    "name": "search_knowledge_base",
                    "description": "Search internal engineering wikis and runbooks.",
                    "parameters": {"query": {"type": "string"}},
                }
            ],
        },
        expected_tool_call={
            "name": None,
            "intent": "answer directly — HTTP is a universal web standard acronym",
        },
        injected_fault={
            "type": "false_tool_trigger",
            "description": "Agent searches the knowledge base for a universally known protocol abbreviation.",
        },
    ),
    Scenario(
        name="S5_simple_percentage",
        user_query="What is 15% of $240?",
        tool_schema={"name": "calculate_metric"},
        env_state={
            "available_tools": [
                {
                    "name": "calculate_metric",
                    "description": "Compute business metrics and KPIs from the data warehouse.",
                    "parameters": {
                        "metric": {"type": "string"},
                        "period": {"type": "string"},
                    },
                }
            ],
        },
        expected_tool_call={
            "name": None,
            "intent": "calculate directly — 15% × $240 = $36",
        },
        injected_fault={
            "type": "false_tool_trigger",
            "description": "Agent calls calculate_metric to compute a trivial percentage that requires no live data.",
        },
    ),
]

_VALIDATORS = {s.name: _no_tool_expected for s in SCENARIOS}


# ---------------------------------------------------------------------------
# Eval Module
# ---------------------------------------------------------------------------

class FalseToolTriggerEval(EvalModule):
    """
    APEX L1 — False Tool Trigger evaluator.

    Scores whether the agent correctly answers questions from general knowledge
    without invoking a tool when no tool call is necessary.

    Score = 1.0  agent answered directly, no tool called
            0.0  agent triggered a tool unnecessarily

    Uses build_l1_agent_fn() from apex.harness.
    """

    failure_mode = "false_tool_trigger"
    layer = FailureLayer.L1_TOOL_SELECTION
    detection_difficulty = DetectionDifficulty.MEDIUM

    def scenarios(self) -> list[Scenario]:
        return SCENARIOS

    def score(self, raw_result: dict[str, Any]) -> float:
        selected_tool = raw_result.get("selected_tool_name", "")
        agent_response = raw_result.get("agent_response", "")
        scenario_name = raw_result.get("scenario_name", "")

        validator = _VALIDATORS.get(scenario_name)
        if validator is None:
            return 0.0

        sc, reason = validator(selected_tool, agent_response, raw_result.get("env_state", {}))
        raw_result["score_reason"] = reason
        return sc

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
        # Agent detects its own false trigger if it notes the tool wasn't needed
        detection_signals = [
            "don't need", "no tool", "general knowledge", "directly",
            "without looking", "i know", "that's simply", "from memory",
        ]
        failure_detected = any(s in agent_response.lower() for s in detection_signals)

        return EvalResult(
            scenario_name=scenario.name,
            failure_mode=self.failure_mode,
            layer=self.layer,
            passed=score >= 1.0,
            score=score,
            detection_difficulty=self.detection_difficulty,
            actual_tool_call={"selected_tool_name": raw_result.get("selected_tool_name", "")},
            agent_response=agent_response,
            failure_detected=failure_detected,
            latency_ms=elapsed,
            metadata={
                "score_reason": raw_result.get("score_reason", ""),
                "injected_fault": scenario.injected_fault,
                "tool_called": raw_result.get("tool_called", False),
            },
        )
