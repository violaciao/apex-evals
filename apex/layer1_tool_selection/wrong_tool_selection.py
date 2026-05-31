"""
apex/layer1_tool_selection/wrong_tool_selection.py
--------------------------------------------------
APEX Layer 1 — Tool Selection
Failure Mode 1.3: Wrong Tool Selection

Definition:
    The agent selects the wrong tool from the available set.
    The chosen tool executes without error but returns data irrelevant
    to the user's intent — because the agent misread the query or
    confused two similarly named tools.

Why it's hard:
    The wrong tool may return valid, plausible data (e.g., querying
    inventory when orders were needed). The final response looks
    reasonable but is semantically incorrect. Only detectable by
    comparing the selected tool against the one required by intent.

Detection difficulty: MEDIUM
    Tool call logs reveal the wrong tool; final response may expose
    the mismatch if the agent uses the result directly.

Evaluation approach:
    1. Present the agent with a query and a set of tools that includes
       one correct tool and at least one plausible but wrong tool
    2. Score: correct tool selected = 1.0, wrong tool = 0.0

Scenarios:
    S1 — Refund statistics       (returns DB vs orders DB)
    S2 — Customer contact        (lookup_customer vs lookup_employee)
    S3 — Shipping ETA            (get_shipment_status vs get_order_details)
    S4 — Product margin          (get_cost_basis vs get_catalog_price)
    S5 — Support ticket          (get_support_ticket vs get_sales_order)
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
# ---------------------------------------------------------------------------

def _make_validator(correct_tool: str, wrong_tools: list[str]):
    """Return a validator that passes iff the agent called correct_tool."""
    def _validate(selected_tool: str, agent_response: str, env: dict) -> tuple[float, str]:
        if selected_tool == correct_tool:
            return 1.0, f"correctly selected '{correct_tool}'"
        if selected_tool in wrong_tools:
            return 0.0, (
                f"selected '{selected_tool}' instead of '{correct_tool}' — "
                "the wrong tool returns unrelated data for this query"
            )
        if not selected_tool:
            return 0.0, f"made no tool call — should have called '{correct_tool}'"
        return 0.0, f"selected '{selected_tool}' — correct tool is '{correct_tool}'"
    return _validate


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="S1_refund_statistics",
        user_query="Show me our refund statistics for April 2026.",
        tool_schema={"name": "multi_tool"},
        env_state={
            "available_tools": [
                {
                    "name": "query_returns_db",
                    "description": "Query the returns and refunds database for refund records, amounts, and reasons.",
                    "parameters": {
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                    },
                },
                {
                    "name": "query_orders_db",
                    "description": "Query the orders database for order counts, revenue, and status.",
                    "parameters": {
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                    },
                },
            ],
            "correct_tool": "query_returns_db",
            "wrong_tools": ["query_orders_db"],
        },
        expected_tool_call={
            "name": "query_returns_db",
            "intent": "refund data lives in the returns DB, not the orders DB",
        },
        injected_fault={
            "type": "wrong_tool_selection",
            "description": "Agent calls query_orders_db which returns order revenue, not refund statistics.",
        },
    ),
    Scenario(
        name="S2_customer_contact",
        user_query="Look up the contact details for customer ID 7701.",
        tool_schema={"name": "multi_tool"},
        env_state={
            "available_tools": [
                {
                    "name": "lookup_customer",
                    "description": "Retrieve a customer record by ID: name, email, phone, and account status.",
                    "parameters": {"customer_id": {"type": "integer"}},
                },
                {
                    "name": "lookup_employee",
                    "description": "Retrieve an employee record by staff ID: name, department, and role.",
                    "parameters": {"employee_id": {"type": "integer"}},
                },
            ],
            "correct_tool": "lookup_customer",
            "wrong_tools": ["lookup_employee"],
        },
        expected_tool_call={
            "name": "lookup_customer",
            "intent": "ID 7701 refers to a customer, not an employee — use lookup_customer",
        },
        injected_fault={
            "type": "wrong_tool_selection",
            "description": "Agent calls lookup_employee with ID 7701, returning an employee record instead of a customer record.",
        },
    ),
    Scenario(
        name="S3_shipping_eta",
        user_query="What is the estimated delivery date for order #5541?",
        tool_schema={"name": "multi_tool"},
        env_state={
            "available_tools": [
                {
                    "name": "get_shipment_status",
                    "description": "Retrieve real-time shipment tracking data including carrier, tracking number, and estimated delivery date.",
                    "parameters": {"order_id": {"type": "integer"}},
                },
                {
                    "name": "get_order_details",
                    "description": "Retrieve order record details: line items, pricing, billing address, and order status.",
                    "parameters": {"order_id": {"type": "integer"}},
                },
            ],
            "correct_tool": "get_shipment_status",
            "wrong_tools": ["get_order_details"],
        },
        expected_tool_call={
            "name": "get_shipment_status",
            "intent": "delivery date is in the shipment record, not the order record",
        },
        injected_fault={
            "type": "wrong_tool_selection",
            "description": "Agent calls get_order_details which returns order line items but not the carrier's estimated delivery date.",
        },
    ),
    Scenario(
        name="S4_product_margin",
        user_query="What is our gross margin on product SKU-B88?",
        tool_schema={"name": "multi_tool"},
        env_state={
            "available_tools": [
                {
                    "name": "get_cost_basis",
                    "description": "Retrieve the cost of goods sold (COGS) and landed cost for a product SKU.",
                    "parameters": {"sku": {"type": "string"}},
                },
                {
                    "name": "get_catalog_price",
                    "description": "Retrieve the current list price and any active discounts for a product SKU.",
                    "parameters": {"sku": {"type": "string"}},
                },
            ],
            "correct_tool": "get_cost_basis",
            "wrong_tools": ["get_catalog_price"],
        },
        expected_tool_call={
            "name": "get_cost_basis",
            "intent": "margin requires COGS data — get_cost_basis provides that; get_catalog_price only has list price",
        },
        injected_fault={
            "type": "wrong_tool_selection",
            "description": "Agent calls get_catalog_price which returns the list price but not COGS, making margin computation impossible.",
        },
    ),
    Scenario(
        name="S5_support_ticket",
        user_query="Pull up the details for support reference TX-2210.",
        tool_schema={"name": "multi_tool"},
        env_state={
            "available_tools": [
                {
                    "name": "get_support_ticket",
                    "description": "Retrieve a customer support ticket by reference number, including issue description, status, and assigned agent.",
                    "parameters": {"reference": {"type": "string"}},
                },
                {
                    "name": "get_sales_order",
                    "description": "Retrieve a sales order by order reference number, including items, pricing, and delivery status.",
                    "parameters": {"order_ref": {"type": "string"}},
                },
            ],
            "correct_tool": "get_support_ticket",
            "wrong_tools": ["get_sales_order"],
        },
        expected_tool_call={
            "name": "get_support_ticket",
            "intent": "TX-2210 is a support ticket reference — use get_support_ticket, not get_sales_order",
        },
        injected_fault={
            "type": "wrong_tool_selection",
            "description": "Agent calls get_sales_order with TX-2210, returning no result or an unrelated order record.",
        },
    ),
]

_VALIDATORS = {
    "S1_refund_statistics": _make_validator("query_returns_db",   ["query_orders_db"]),
    "S2_customer_contact":  _make_validator("lookup_customer",    ["lookup_employee"]),
    "S3_shipping_eta":      _make_validator("get_shipment_status", ["get_order_details"]),
    "S4_product_margin":    _make_validator("get_cost_basis",     ["get_catalog_price"]),
    "S5_support_ticket":    _make_validator("get_support_ticket", ["get_sales_order"]),
}


# ---------------------------------------------------------------------------
# Eval Module
# ---------------------------------------------------------------------------

class WrongToolSelectionEval(EvalModule):
    """
    APEX L1 — Wrong Tool Selection evaluator.

    Scores whether the agent selects the correct tool from a set that
    includes at least one plausible but semantically wrong alternative.

    Score = 1.0  agent called the correct tool
            0.0  agent called the wrong tool, a different tool, or no tool

    Uses build_l1_agent_fn() from apex.harness.
    """

    failure_mode = "wrong_tool_selection"
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
        detection_signals = [
            "wrong tool", "incorrect tool", "should use", "meant to call",
            "different tool", "not the right",
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
