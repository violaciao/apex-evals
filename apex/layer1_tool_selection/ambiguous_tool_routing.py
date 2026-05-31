"""
apex/layer1_tool_selection/ambiguous_tool_routing.py
----------------------------------------------------
APEX Layer 1 — Tool Selection
Failure Mode 1.4: Ambiguous Tool Routing

Definition:
    The user's request maps plausibly to multiple tools. The agent
    picks one without resolving the ambiguity — and picks the wrong one
    for the given context. The correct tool is determinable from context
    clues present in the conversation.

Why it's hard:
    Every tool in the set is a reasonable guess. The query is legitimately
    underspecified in isolation. The correct answer requires reading
    contextual signals (department, role, prior topic) that the agent
    must attend to — not just matching query keywords to tool names.

Detection difficulty: MEDIUM-HIGH
    The chosen tool may return valid data, but for the wrong semantic
    scope. Requires intent-level analysis to detect. The agent may not
    recognize the ambiguity at all.

Evaluation approach:
    1. Present agent with a query that maps to multiple available tools
    2. Provide context that resolves the ambiguity to one correct tool
    3. Score: correct tool = 1.0; clarification request = 0.5 (ambiguity
       acknowledged but not resolved); wrong tool = 0.0

Scenarios:
    S1 — "Pull up the account"   (billing vs sales — context: invoice dispute)
    S2 — "Get the report"        (inventory vs sales — context: warehouse manager)
    S3 — "Update the record"     (customer vs order — context: customer changed email)
    S4 — "Check the status"      (order vs ticket — context: undelivered package)
    S5 — "Show account activity" (bank feed vs user activity — context: finance reconciliation)
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

_CLARIFICATION_SIGNALS = [
    "which", "clarif", "do you mean", "specify", "could you", "what kind",
    "are you referring", "please confirm", "ambiguous",
]


# ---------------------------------------------------------------------------
# Validators: pure Python
# 1.0 = correct tool; 0.5 = asked for clarification; 0.0 = wrong routing
# ---------------------------------------------------------------------------

def _make_validator(correct_tool: str, wrong_tools: list[str]):
    def _validate(selected_tool: str, agent_response: str, env: dict) -> tuple[float, str]:
        if selected_tool == correct_tool:
            return 1.0, f"correctly routed to '{correct_tool}' based on context"
        asked_for_clarification = (
            not selected_tool
            and any(s in agent_response.lower() for s in _CLARIFICATION_SIGNALS)
        )
        if asked_for_clarification:
            return 0.5, "asked for clarification — ambiguity acknowledged but context was sufficient to resolve"
        if selected_tool in wrong_tools:
            return 0.0, (
                f"routed to '{selected_tool}' — context indicated '{correct_tool}' was required"
            )
        if not selected_tool:
            return 0.0, f"made no tool call — should have routed to '{correct_tool}'"
        return 0.0, f"routed to '{selected_tool}' — correct was '{correct_tool}'"
    return _validate


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="S1_account_billing_vs_sales",
        user_query="Pull up the account for Acme Corp.",
        tool_schema={"name": "multi_tool"},
        env_state={
            "context": "The billing team is investigating a disputed invoice from Acme Corp. They need the billing account details.",
            "available_tools": [
                {
                    "name": "get_billing_account",
                    "description": "Retrieve a customer's billing account: invoices, payment history, and outstanding balance.",
                    "parameters": {"company_name": {"type": "string"}},
                },
                {
                    "name": "get_sales_account",
                    "description": "Retrieve a customer's sales account: deals, pipeline, and assigned sales rep.",
                    "parameters": {"company_name": {"type": "string"}},
                },
            ],
            "correct_tool": "get_billing_account",
            "wrong_tools": ["get_sales_account"],
        },
        expected_tool_call={
            "name": "get_billing_account",
            "intent": "context says 'billing team' and 'disputed invoice' — billing account is correct",
        },
        injected_fault={
            "type": "ambiguous_tool_routing",
            "description": "Agent calls get_sales_account despite context clearly indicating a billing dispute.",
        },
    ),
    Scenario(
        name="S2_report_inventory_vs_sales",
        user_query="Get the report for last week.",
        tool_schema={"name": "multi_tool"},
        env_state={
            "context": "A warehouse manager is reviewing stock levels to determine which SKUs need to be reordered.",
            "available_tools": [
                {
                    "name": "get_inventory_report",
                    "description": "Generate a weekly inventory report showing stock levels, low-stock alerts, and reorder suggestions.",
                    "parameters": {"week_ending": {"type": "string"}},
                },
                {
                    "name": "get_sales_report",
                    "description": "Generate a weekly sales report showing revenue, order volume, and top-selling products.",
                    "parameters": {"week_ending": {"type": "string"}},
                },
            ],
            "correct_tool": "get_inventory_report",
            "wrong_tools": ["get_sales_report"],
        },
        expected_tool_call={
            "name": "get_inventory_report",
            "intent": "context says 'warehouse manager' reviewing 'stock levels' and 'reorder' — inventory report is correct",
        },
        injected_fault={
            "type": "ambiguous_tool_routing",
            "description": "Agent calls get_sales_report, returning revenue data irrelevant to the warehouse manager's reorder task.",
        },
    ),
    Scenario(
        name="S3_update_customer_vs_order",
        user_query="Update the record with the new phone number.",
        tool_schema={"name": "multi_tool"},
        env_state={
            "context": "Customer Sarah Chen (ID 9021) called in to update her contact information after moving.",
            "available_tools": [
                {
                    "name": "update_customer_record",
                    "description": "Update a customer's profile fields: name, email, phone, or billing address.",
                    "parameters": {
                        "customer_id": {"type": "integer"},
                        "field": {"type": "string"},
                        "value": {"type": "string"},
                    },
                },
                {
                    "name": "update_order_record",
                    "description": "Update a field on an existing order: shipping address, delivery notes, or priority flag.",
                    "parameters": {
                        "order_id": {"type": "integer"},
                        "field": {"type": "string"},
                        "value": {"type": "string"},
                    },
                },
            ],
            "correct_tool": "update_customer_record",
            "wrong_tools": ["update_order_record"],
        },
        expected_tool_call={
            "name": "update_customer_record",
            "intent": "context identifies a customer (Sarah Chen, ID 9021) updating contact info — customer record, not an order",
        },
        injected_fault={
            "type": "ambiguous_tool_routing",
            "description": "Agent calls update_order_record to update a phone number, which is a customer profile field, not an order field.",
        },
    ),
    Scenario(
        name="S4_status_order_vs_ticket",
        user_query="Can you check the status for me?",
        tool_schema={"name": "multi_tool"},
        env_state={
            "context": "A customer says their package was supposed to arrive three days ago and wants to know where it is.",
            "available_tools": [
                {
                    "name": "get_shipment_status",
                    "description": "Check the real-time carrier tracking status and estimated delivery for an order.",
                    "parameters": {"order_id": {"type": "integer"}},
                },
                {
                    "name": "get_ticket_status",
                    "description": "Check the status of a customer support ticket: open, in-progress, or resolved.",
                    "parameters": {"ticket_id": {"type": "string"}},
                },
            ],
            "correct_tool": "get_shipment_status",
            "wrong_tools": ["get_ticket_status"],
        },
        expected_tool_call={
            "name": "get_shipment_status",
            "intent": "context is a missing package — shipment tracking is needed, not a support ticket status",
        },
        injected_fault={
            "type": "ambiguous_tool_routing",
            "description": "Agent calls get_ticket_status when the customer wants shipment tracking, not a support ticket status.",
        },
    ),
    Scenario(
        name="S5_activity_bank_vs_user",
        user_query="Show me the activity for this account.",
        tool_schema={"name": "multi_tool"},
        env_state={
            "context": "The finance team needs to reconcile last month's transactions against the bank feed.",
            "available_tools": [
                {
                    "name": "get_bank_activity",
                    "description": "Retrieve bank transaction feed for a given account and date range for reconciliation.",
                    "parameters": {
                        "account_id": {"type": "string"},
                        "month": {"type": "string"},
                    },
                },
                {
                    "name": "get_user_activity",
                    "description": "Retrieve a user's in-app activity log: logins, feature usage, and session history.",
                    "parameters": {"user_id": {"type": "string"}},
                },
            ],
            "correct_tool": "get_bank_activity",
            "wrong_tools": ["get_user_activity"],
        },
        expected_tool_call={
            "name": "get_bank_activity",
            "intent": "context says 'finance team', 'reconcile', 'bank feed' — bank transaction activity is correct",
        },
        injected_fault={
            "type": "ambiguous_tool_routing",
            "description": "Agent calls get_user_activity returning in-app session logs instead of the bank transaction feed needed for reconciliation.",
        },
    ),
]

_VALIDATORS = {
    "S1_account_billing_vs_sales":  _make_validator("get_billing_account",  ["get_sales_account"]),
    "S2_report_inventory_vs_sales": _make_validator("get_inventory_report", ["get_sales_report"]),
    "S3_update_customer_vs_order":  _make_validator("update_customer_record", ["update_order_record"]),
    "S4_status_order_vs_ticket":    _make_validator("get_shipment_status",  ["get_ticket_status"]),
    "S5_activity_bank_vs_user":     _make_validator("get_bank_activity",    ["get_user_activity"]),
}


# ---------------------------------------------------------------------------
# Eval Module
# ---------------------------------------------------------------------------

class AmbiguousToolRoutingEval(EvalModule):
    """
    APEX L1 — Ambiguous Tool Routing evaluator.

    Scores whether the agent correctly resolves routing ambiguity using
    contextual signals when a query maps plausibly to multiple tools.

    Score = 1.0  agent routed to the correct tool given context
            0.5  agent asked for clarification (ambiguity acknowledged)
            0.0  agent routed to the wrong tool or made no call

    Uses build_l1_agent_fn() from apex.harness.
    """

    failure_mode = "ambiguous_tool_routing"
    layer = FailureLayer.L1_TOOL_SELECTION
    detection_difficulty = DetectionDifficulty.MEDIUM_HIGH

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
        failure_detected = any(s in agent_response.lower() for s in _CLARIFICATION_SIGNALS)

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
