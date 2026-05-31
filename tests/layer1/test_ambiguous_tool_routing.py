"""
tests/layer1/test_ambiguous_tool_routing.py
---------------------------------------------
APEX Layer 1 — Ambiguous Tool Routing
pytest suite: 5 scenarios × context-driven routing validation

Run (zero API cost — validator unit tests):
    pytest tests/layer1/test_ambiguous_tool_routing.py -v -k "not live"

Run (live LLM):
    APEX_PROFILE=free pytest tests/layer1/test_ambiguous_tool_routing.py -v -m free -s
"""

import pytest
from apex.layer1_tool_selection.ambiguous_tool_routing import (
    AmbiguousToolRoutingEval,
    SCENARIOS,
    _make_validator,
    _CLARIFICATION_SIGNALS,
)
from apex.base import FailureLayer, DetectionDifficulty


# ---------------------------------------------------------------------------
# Unit tests: validators (zero LLM cost)
# ---------------------------------------------------------------------------

class TestValidators:

    def setup_method(self):
        self.v_billing   = _make_validator("get_billing_account",   ["get_sales_account"])
        self.v_inventory = _make_validator("get_inventory_report",  ["get_sales_report"])
        self.v_customer  = _make_validator("update_customer_record", ["update_order_record"])
        self.v_shipment  = _make_validator("get_shipment_status",   ["get_ticket_status"])
        self.v_bank      = _make_validator("get_bank_activity",     ["get_user_activity"])

    # S1 — billing vs sales
    def test_s1_correct_routing_passes(self):
        score, reason = self.v_billing("get_billing_account", "", {})
        assert score == 1.0
        assert "get_billing_account" in reason

    def test_s1_wrong_routing_fails(self):
        score, reason = self.v_billing("get_sales_account", "", {})
        assert score == 0.0
        assert "get_billing_account" in reason

    def test_s1_clarification_gives_half_credit(self):
        score, reason = self.v_billing("", "Which account do you mean — billing or sales?", {})
        assert score == 0.5
        assert "clarif" in reason.lower() or "acknowledge" in reason.lower()

    def test_s1_no_call_no_clarification_fails(self):
        score, reason = self.v_billing("", "I can help with that.", {})
        assert score == 0.0

    # S2 — inventory vs sales report
    def test_s2_correct_routing_passes(self):
        score, _ = self.v_inventory("get_inventory_report", "", {})
        assert score == 1.0

    def test_s2_wrong_routing_fails(self):
        score, reason = self.v_inventory("get_sales_report", "", {})
        assert score == 0.0
        assert "get_inventory_report" in reason

    # S3 — customer vs order update
    def test_s3_correct_routing_passes(self):
        score, _ = self.v_customer("update_customer_record", "", {})
        assert score == 1.0

    def test_s3_wrong_routing_fails(self):
        score, reason = self.v_customer("update_order_record", "", {})
        assert score == 0.0
        assert "update_customer_record" in reason

    # S4 — shipment vs ticket status
    def test_s4_correct_routing_passes(self):
        score, _ = self.v_shipment("get_shipment_status", "", {})
        assert score == 1.0

    def test_s4_wrong_routing_fails(self):
        score, reason = self.v_shipment("get_ticket_status", "", {})
        assert score == 0.0

    # S5 — bank vs user activity
    def test_s5_correct_routing_passes(self):
        score, _ = self.v_bank("get_bank_activity", "", {})
        assert score == 1.0

    def test_s5_wrong_routing_fails(self):
        score, reason = self.v_bank("get_user_activity", "", {})
        assert score == 0.0
        assert "get_bank_activity" in reason


# ---------------------------------------------------------------------------
# Module tests: structure & metadata (zero LLM cost)
# ---------------------------------------------------------------------------

class TestEvalModuleStructure:

    def setup_method(self):
        self.module = AmbiguousToolRoutingEval()

    def test_failure_mode_identifier(self):
        assert self.module.failure_mode == "ambiguous_tool_routing"

    def test_layer_assignment(self):
        assert self.module.layer == FailureLayer.L1_TOOL_SELECTION

    def test_detection_difficulty(self):
        assert self.module.detection_difficulty == DetectionDifficulty.MEDIUM_HIGH

    def test_scenarios_count(self):
        assert len(self.module.scenarios()) == 5

    def test_all_scenarios_have_injected_fault(self):
        for s in self.module.scenarios():
            assert s.injected_fault, f"{s.name} missing injected_fault"
            assert "type" in s.injected_fault and "description" in s.injected_fault

    def test_all_scenarios_have_context(self):
        """Each scenario must provide context that resolves the ambiguity."""
        for s in self.module.scenarios():
            ctx = s.env_state.get("context", "")
            assert ctx, f"{s.name}: context is required to resolve routing ambiguity"

    def test_all_scenarios_have_two_plausible_tools(self):
        for s in self.module.scenarios():
            tools = s.env_state.get("available_tools", [])
            assert len(tools) == 2, f"{s.name}: must have exactly 2 plausible tools"

    def test_correct_and_wrong_tools_specified(self):
        for s in self.module.scenarios():
            assert s.env_state.get("correct_tool"), f"{s.name}: must specify correct_tool"
            assert s.env_state.get("wrong_tools"), f"{s.name}: must specify wrong_tools"

    def test_query_is_ambiguous_without_context(self):
        """Each query should be short and genuinely ambiguous in isolation."""
        for s in self.module.scenarios():
            assert len(s.user_query) < 80, (
                f"{s.name}: query should be short/ambiguous — got {len(s.user_query)} chars"
            )

    def test_injected_fault_type(self):
        for s in self.module.scenarios():
            assert s.injected_fault["type"] == "ambiguous_tool_routing"


# ---------------------------------------------------------------------------
# Scorer unit tests (zero LLM cost)
# ---------------------------------------------------------------------------

class TestScorer:

    def setup_method(self):
        self.module = AmbiguousToolRoutingEval()

    def test_score_correct_tool_returns_1(self):
        raw = {
            "scenario_name": "S1_account_billing_vs_sales",
            "selected_tool_name": "get_billing_account",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_wrong_tool_returns_0(self):
        raw = {
            "scenario_name": "S1_account_billing_vs_sales",
            "selected_tool_name": "get_sales_account",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_clarification_returns_half(self):
        raw = {
            "scenario_name": "S2_report_inventory_vs_sales",
            "selected_tool_name": "",
            "agent_response": "Could you clarify which type of report you need?",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.5

    def test_score_no_call_no_clarification_returns_0(self):
        raw = {
            "scenario_name": "S3_update_customer_vs_order",
            "selected_tool_name": "",
            "agent_response": "I'll help with that.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_correct_tool_s5(self):
        raw = {
            "scenario_name": "S5_activity_bank_vs_user",
            "selected_tool_name": "get_bank_activity",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_unknown_scenario_returns_0(self):
        raw = {
            "scenario_name": "S99_unknown",
            "selected_tool_name": "get_billing_account",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0


# ---------------------------------------------------------------------------
# Integration tests: live LLM (requires APEX_PROFILE + API key)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.free
async def test_live_s4_status_check():
    """
    Live: S4 shipment vs ticket status — checks context-driven routing.

    Run: APEX_PROFILE=free pytest tests/layer1/test_ambiguous_tool_routing.py::test_live_s4_status_check -v -s
    """
    from apex.harness import build_l1_agent_fn

    module = AmbiguousToolRoutingEval()
    agent_fn = build_l1_agent_fn(verbose=True)

    scenario = next(s for s in module.scenarios() if s.name == "S4_status_order_vs_ticket")
    result = await module.run(scenario, agent_fn)

    print(f"\n{result.summary()}")
    print(f"Tool selected: {result.actual_tool_call}")
    print(f"Reason:        {result.metadata.get('score_reason')}")

    assert result.score in (0.0, 0.5, 1.0)
    assert result.latency_ms > 0


@pytest.mark.asyncio
@pytest.mark.free
async def test_live_all_scenarios():
    """
    Full suite: all 5 ambiguous routing scenarios against the active LLM.

    Run: APEX_PROFILE=free pytest tests/layer1/test_ambiguous_tool_routing.py::test_live_all_scenarios -v -s
    """
    from apex.harness import build_l1_agent_fn

    module = AmbiguousToolRoutingEval()
    agent_fn = build_l1_agent_fn(verbose=False)
    results = await module.run_all(agent_fn)

    print("\n" + "=" * 60)
    print("APEX L1 — Ambiguous Tool Routing Results")
    print("=" * 60)
    for r in results:
        print(r.summary())
        print(f"  tool_selected: {r.actual_tool_call}")
        print(f"  score:         {r.score}")
        print(f"  reason:        {r.metadata.get('score_reason')}")

    avg_score = sum(r.score for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.2f} / 1.00")
    print(f"Pass rate (>=1.0): {sum(r.passed for r in results)}/{len(results)}")
    partial = sum(1 for r in results if r.score == 0.5)
    print(f"Partial (0.5):     {partial}/{len(results)}")

    assert len(results) == 5
