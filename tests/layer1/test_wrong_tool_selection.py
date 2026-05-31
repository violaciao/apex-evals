"""
tests/layer1/test_wrong_tool_selection.py
------------------------------------------
APEX Layer 1 — Wrong Tool Selection
pytest suite: 5 scenarios × correct-tool discrimination

Run (zero API cost — validator unit tests):
    pytest tests/layer1/test_wrong_tool_selection.py -v -k "not live"

Run (live LLM):
    APEX_PROFILE=free pytest tests/layer1/test_wrong_tool_selection.py -v -m free -s
"""

import pytest
from apex.layer1_tool_selection.wrong_tool_selection import (
    WrongToolSelectionEval,
    SCENARIOS,
    _make_validator,
)
from apex.base import FailureLayer, DetectionDifficulty


# ---------------------------------------------------------------------------
# Unit tests: validators (zero LLM cost)
# ---------------------------------------------------------------------------

class TestValidators:

    def setup_method(self):
        self.v_refund  = _make_validator("query_returns_db",    ["query_orders_db"])
        self.v_customer = _make_validator("lookup_customer",    ["lookup_employee"])
        self.v_shipping = _make_validator("get_shipment_status", ["get_order_details"])
        self.v_margin   = _make_validator("get_cost_basis",     ["get_catalog_price"])
        self.v_ticket   = _make_validator("get_support_ticket", ["get_sales_order"])

    # S1 — Refund statistics
    def test_s1_correct_tool_passes(self):
        score, _ = self.v_refund("query_returns_db", "", {})
        assert score == 1.0

    def test_s1_wrong_tool_fails(self):
        score, reason = self.v_refund("query_orders_db", "", {})
        assert score == 0.0
        assert "query_returns_db" in reason

    def test_s1_no_tool_fails(self):
        score, reason = self.v_refund("", "", {})
        assert score == 0.0
        assert "query_returns_db" in reason

    # S2 — Customer contact
    def test_s2_correct_tool_passes(self):
        score, _ = self.v_customer("lookup_customer", "", {})
        assert score == 1.0

    def test_s2_wrong_tool_fails(self):
        score, reason = self.v_customer("lookup_employee", "", {})
        assert score == 0.0
        assert "lookup_customer" in reason

    # S3 — Shipping ETA
    def test_s3_correct_tool_passes(self):
        score, _ = self.v_shipping("get_shipment_status", "", {})
        assert score == 1.0

    def test_s3_wrong_tool_fails(self):
        score, reason = self.v_shipping("get_order_details", "", {})
        assert score == 0.0
        assert "get_shipment_status" in reason

    # S4 — Product margin
    def test_s4_correct_tool_passes(self):
        score, _ = self.v_margin("get_cost_basis", "", {})
        assert score == 1.0

    def test_s4_wrong_tool_fails(self):
        score, reason = self.v_margin("get_catalog_price", "", {})
        assert score == 0.0
        assert "get_cost_basis" in reason

    # S5 — Support ticket
    def test_s5_correct_tool_passes(self):
        score, _ = self.v_ticket("get_support_ticket", "", {})
        assert score == 1.0

    def test_s5_wrong_tool_fails(self):
        score, reason = self.v_ticket("get_sales_order", "", {})
        assert score == 0.0
        assert "get_support_ticket" in reason


# ---------------------------------------------------------------------------
# Module tests: structure & metadata (zero LLM cost)
# ---------------------------------------------------------------------------

class TestEvalModuleStructure:

    def setup_method(self):
        self.module = WrongToolSelectionEval()

    def test_failure_mode_identifier(self):
        assert self.module.failure_mode == "wrong_tool_selection"

    def test_layer_assignment(self):
        assert self.module.layer == FailureLayer.L1_TOOL_SELECTION

    def test_detection_difficulty(self):
        assert self.module.detection_difficulty == DetectionDifficulty.MEDIUM

    def test_scenarios_count(self):
        assert len(self.module.scenarios()) == 5

    def test_all_scenarios_have_injected_fault(self):
        for s in self.module.scenarios():
            assert s.injected_fault, f"{s.name} missing injected_fault"
            assert "type" in s.injected_fault and "description" in s.injected_fault

    def test_all_scenarios_have_correct_and_wrong_tool(self):
        for s in self.module.scenarios():
            tools = s.env_state.get("available_tools", [])
            assert len(tools) >= 2, f"{s.name}: must have at least one correct and one wrong tool"
            correct = s.env_state.get("correct_tool")
            wrong = s.env_state.get("wrong_tools", [])
            assert correct, f"{s.name}: env_state must specify correct_tool"
            assert wrong, f"{s.name}: env_state must specify wrong_tools"

    def test_correct_tool_is_in_available_tools(self):
        for s in self.module.scenarios():
            tool_names = {t["name"] for t in s.env_state.get("available_tools", [])}
            correct = s.env_state.get("correct_tool")
            assert correct in tool_names, (
                f"{s.name}: correct_tool '{correct}' must be in available_tools"
            )

    def test_wrong_tools_are_in_available_tools(self):
        for s in self.module.scenarios():
            tool_names = {t["name"] for t in s.env_state.get("available_tools", [])}
            for w in s.env_state.get("wrong_tools", []):
                assert w in tool_names, (
                    f"{s.name}: wrong_tool '{w}' must be in available_tools"
                )

    def test_injected_fault_type(self):
        for s in self.module.scenarios():
            assert s.injected_fault["type"] == "wrong_tool_selection"


# ---------------------------------------------------------------------------
# Scorer unit tests (zero LLM cost)
# ---------------------------------------------------------------------------

class TestScorer:

    def setup_method(self):
        self.module = WrongToolSelectionEval()

    def test_score_correct_tool_returns_1(self):
        raw = {
            "scenario_name": "S1_refund_statistics",
            "selected_tool_name": "query_returns_db",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_wrong_tool_returns_0(self):
        raw = {
            "scenario_name": "S1_refund_statistics",
            "selected_tool_name": "query_orders_db",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_no_tool_returns_0(self):
        raw = {
            "scenario_name": "S2_customer_contact",
            "selected_tool_name": "",
            "agent_response": "I cannot look that up.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_correct_tool_s4(self):
        raw = {
            "scenario_name": "S4_product_margin",
            "selected_tool_name": "get_cost_basis",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_unknown_scenario_returns_0(self):
        raw = {
            "scenario_name": "S99_unknown",
            "selected_tool_name": "some_tool",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0


# ---------------------------------------------------------------------------
# Integration tests: live LLM (requires APEX_PROFILE + API key)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.free
async def test_live_s2_customer_contact():
    """
    Live: S2 customer vs employee — checks whether agent picks lookup_customer or lookup_employee.

    Run: APEX_PROFILE=free pytest tests/layer1/test_wrong_tool_selection.py::test_live_s2_customer_contact -v -s
    """
    from apex.harness import build_l1_agent_fn

    module = WrongToolSelectionEval()
    agent_fn = build_l1_agent_fn(verbose=True)

    scenario = next(s for s in module.scenarios() if s.name == "S2_customer_contact")
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
    Full suite: all 5 wrong-tool-selection scenarios against the active LLM.

    Run: APEX_PROFILE=free pytest tests/layer1/test_wrong_tool_selection.py::test_live_all_scenarios -v -s
    """
    from apex.harness import build_l1_agent_fn

    module = WrongToolSelectionEval()
    agent_fn = build_l1_agent_fn(verbose=False)
    results = await module.run_all(agent_fn)

    print("\n" + "=" * 60)
    print("APEX L1 — Wrong Tool Selection Results")
    print("=" * 60)
    for r in results:
        print(r.summary())
        print(f"  tool_selected: {r.actual_tool_call}")
        print(f"  reason:        {r.metadata.get('score_reason')}")

    avg_score = sum(r.score for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.2f} / 1.00")
    print(f"Pass rate: {sum(r.passed for r in results)}/{len(results)}")

    assert len(results) == 5
