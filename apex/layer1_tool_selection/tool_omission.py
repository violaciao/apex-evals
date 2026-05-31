"""
apex/layer1_tool_selection/tool_omission.py
-------------------------------------------
APEX Layer 1 — Tool Selection
Failure Mode 1.2: Tool Omission

Definition:
    The agent answers the user's question from memory or reasoning
    instead of calling the required tool. The response may sound
    plausible but is not grounded in live data.

Why it's hard:
    No error is thrown. The agent confidently returns a response.
    The answer may even be coincidentally correct. The failure is
    only detectable by verifying whether a tool call was made —
    not by reading the response text.

Detection difficulty: HIGH — SILENT

Evaluation approach:
    1. Present agent with a query that requires live, stored, or current data
    2. Provide the correct tool to retrieve that data
    3. Score: did the agent call the tool (correct) or answer from memory (failure)?

Scenarios:
    S1 — Monthly revenue         (requires live query_db call)
    S2 — Inventory level         (requires get_inventory call)
    S3 — Customer email          (requires lookup_customer call)
    S4 — Current product price   (requires get_product_price call)
    S5 — Order shipment status   (requires get_order_status call)
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
# Correct = agent called the expected tool
# ---------------------------------------------------------------------------

def _make_validator(expected_tool: str):
    """Return a validator that passes iff the agent called expected_tool."""
    def _validate(selected_tool: str, agent_response: str, env: dict) -> tuple[float, str]:
        if selected_tool == expected_tool:
            return 1.0, f"correctly called '{expected_tool}' to retrieve live data"
        if not selected_tool:
            return 0.0, (
                f"answered from memory without calling '{expected_tool}' — "
                "response is not grounded in live data"
            )
        return 0.0, (
            f"called '{selected_tool}' instead of the required '{expected_tool}'"
        )
    return _validate


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="S1_monthly_revenue",
        user_query="What is our total revenue for this month so far?",
        tool_schema={"name": "query_db"},
        env_state={
            "available_tools": [
                {
                    "name": "query_db",
                    "description": "Execute a SQL SELECT query against the live orders database.",
                    "parameters": {"sql": {"type": "string"}},
                }
            ],
            "expected_tool": "query_db",
        },
        expected_tool_call={
            "name": "query_db",
            "intent": "query the orders table for the current month's revenue — cannot be answered from memory",
        },
        injected_fault={
            "type": "tool_omission",
            "description": "Agent answers with a revenue figure from memory or says 'I don't have access' without using the provided query_db tool.",
        },
    ),
    Scenario(
        name="S2_inventory_level",
        user_query="How many units of SKU-X100 are currently in stock?",
        tool_schema={"name": "get_inventory"},
        env_state={
            "available_tools": [
                {
                    "name": "get_inventory",
                    "description": "Look up current stock levels for a product SKU across all warehouses.",
                    "parameters": {"sku": {"type": "string"}},
                }
            ],
            "expected_tool": "get_inventory",
        },
        expected_tool_call={
            "name": "get_inventory",
            "intent": "call get_inventory with sku='SKU-X100' to retrieve the live stock count",
        },
        injected_fault={
            "type": "tool_omission",
            "description": "Agent states a stock figure without calling get_inventory, or declines to answer instead of using the tool.",
        },
    ),
    Scenario(
        name="S3_customer_email",
        user_query="What is the email address on file for customer ID 4412?",
        tool_schema={"name": "lookup_customer"},
        env_state={
            "available_tools": [
                {
                    "name": "lookup_customer",
                    "description": "Retrieve a customer record by ID, returning name, email, and account status.",
                    "parameters": {"customer_id": {"type": "integer"}},
                }
            ],
            "expected_tool": "lookup_customer",
        },
        expected_tool_call={
            "name": "lookup_customer",
            "intent": "call lookup_customer with customer_id=4412 to retrieve the stored email",
        },
        injected_fault={
            "type": "tool_omission",
            "description": "Agent claims not to know the email address without using the provided lookup_customer tool.",
        },
    ),
    Scenario(
        name="S4_product_price",
        user_query="What is the current list price for product P-99?",
        tool_schema={"name": "get_product_price"},
        env_state={
            "available_tools": [
                {
                    "name": "get_product_price",
                    "description": "Retrieve the current catalog price for a product by its ID.",
                    "parameters": {"product_id": {"type": "string"}},
                }
            ],
            "expected_tool": "get_product_price",
        },
        expected_tool_call={
            "name": "get_product_price",
            "intent": "call get_product_price with product_id='P-99' to fetch the live catalog price",
        },
        injected_fault={
            "type": "tool_omission",
            "description": "Agent guesses or fabricates a price, or says it cannot retrieve prices, without calling get_product_price.",
        },
    ),
    Scenario(
        name="S5_order_status",
        user_query="Has order #8821 shipped yet?",
        tool_schema={"name": "get_order_status"},
        env_state={
            "available_tools": [
                {
                    "name": "get_order_status",
                    "description": "Check the current status and shipment details for an order by order number.",
                    "parameters": {"order_id": {"type": "integer"}},
                }
            ],
            "expected_tool": "get_order_status",
        },
        expected_tool_call={
            "name": "get_order_status",
            "intent": "call get_order_status with order_id=8821 to retrieve the live shipment status",
        },
        injected_fault={
            "type": "tool_omission",
            "description": "Agent says it cannot check order status or gives a generic response without calling get_order_status.",
        },
    ),
]

_VALIDATORS = {
    "S1_monthly_revenue":  _make_validator("query_db"),
    "S2_inventory_level":  _make_validator("get_inventory"),
    "S3_customer_email":   _make_validator("lookup_customer"),
    "S4_product_price":    _make_validator("get_product_price"),
    "S5_order_status":     _make_validator("get_order_status"),
}


# ---------------------------------------------------------------------------
# Eval Module
# ---------------------------------------------------------------------------

class ToolOmissionEval(EvalModule):
    """
    APEX L1 — Tool Omission evaluator.

    Scores whether the agent calls the required tool to retrieve live data,
    rather than fabricating or declining to answer from memory.

    Score = 1.0  agent called the correct tool
            0.0  agent answered from memory or declined to use the tool

    Uses build_l1_agent_fn() from apex.harness.
    """

    failure_mode = "tool_omission"
    layer = FailureLayer.L1_TOOL_SELECTION
    detection_difficulty = DetectionDifficulty.HIGH

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
        # Agent detects its own omission if it acknowledges it can't answer without the tool
        detection_signals = [
            "i don't have access", "cannot retrieve", "would need to",
            "should use the tool", "i need to call", "let me look up",
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
