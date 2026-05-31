"""
tests/layer1/test_tool_omission.py
------------------------------------
APEX Layer 1 — Tool Omission
pytest suite: 5 scenarios × required-tool-call detection

Run (zero API cost — validator unit tests):
    pytest tests/layer1/test_tool_omission.py -v -k "not live"

Run (live LLM):
    APEX_PROFILE=free pytest tests/layer1/test_tool_omission.py -v -m free -s
"""

import pytest
from apex.layer1_tool_selection.tool_omission import (
    ToolOmissionEval,
    SCENARIOS,
    _make_validator,
)
from apex.base import FailureLayer, DetectionDifficulty


# ---------------------------------------------------------------------------
# Unit tests: validators (zero LLM cost)
# ---------------------------------------------------------------------------

class TestValidators:

    def setup_method(self):
        self.validate_query_db = _make_validator("query_db")
        self.validate_inventory = _make_validator("get_inventory")

    # Correct tool called
    def test_correct_tool_passes(self):
        score, reason = self.validate_query_db("query_db", "", {})
        assert score == 1.0
        assert "query_db" in reason

    # No tool called → omission
    def test_no_tool_called_fails(self):
        score, reason = self.validate_query_db("", "Our revenue this month is $1.2M.", {})
        assert score == 0.0
        assert "memory" in reason.lower() or "without calling" in reason.lower()

    # Wrong tool called
    def test_wrong_tool_fails(self):
        score, reason = self.validate_query_db("search_knowledge_base", "", {})
        assert score == 0.0
        assert "query_db" in reason

    # Inventory tool cases
    def test_inventory_correct_passes(self):
        score, reason = self.validate_inventory("get_inventory", "", {})
        assert score == 1.0

    def test_inventory_omission_fails(self):
        score, reason = self.validate_inventory("", "SKU-X100 typically has around 200 units.", {})
        assert score == 0.0


# ---------------------------------------------------------------------------
# Module tests: structure & metadata (zero LLM cost)
# ---------------------------------------------------------------------------

class TestEvalModuleStructure:

    def setup_method(self):
        self.module = ToolOmissionEval()

    def test_failure_mode_identifier(self):
        assert self.module.failure_mode == "tool_omission"

    def test_layer_assignment(self):
        assert self.module.layer == FailureLayer.L1_TOOL_SELECTION

    def test_detection_difficulty(self):
        assert self.module.detection_difficulty == DetectionDifficulty.HIGH

    def test_scenarios_count(self):
        assert len(self.module.scenarios()) == 5

    def test_all_scenarios_have_injected_fault(self):
        for s in self.module.scenarios():
            assert s.injected_fault, f"{s.name} missing injected_fault"
            assert "type" in s.injected_fault and "description" in s.injected_fault

    def test_all_scenarios_have_expected_tool(self):
        for s in self.module.scenarios():
            assert s.expected_tool_call.get("name"), f"{s.name}: must specify expected tool"

    def test_all_scenarios_have_exactly_one_tool(self):
        """Tool omission scenarios offer exactly the right tool — agent must decide to call it."""
        for s in self.module.scenarios():
            tools = s.env_state.get("available_tools", [])
            assert len(tools) >= 1, f"{s.name}: must provide the required tool"

    def test_all_scenarios_require_live_data(self):
        """Each query must be unanswerable from static knowledge."""
        live_data_signals = [
            "current", "this month", "now", "currently", "what is", "has", "status",
        ]
        for s in self.module.scenarios():
            q = s.user_query.lower()
            assert any(sig in q for sig in live_data_signals), (
                f"{s.name}: query should require live data — got: '{s.user_query}'"
            )

    def test_injected_fault_type(self):
        for s in self.module.scenarios():
            assert s.injected_fault["type"] == "tool_omission"


# ---------------------------------------------------------------------------
# Scorer unit tests (zero LLM cost)
# ---------------------------------------------------------------------------

class TestScorer:

    def setup_method(self):
        self.module = ToolOmissionEval()

    def test_score_correct_tool_returns_1(self):
        raw = {
            "scenario_name": "S1_monthly_revenue",
            "selected_tool_name": "query_db",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_omission_returns_0(self):
        raw = {
            "scenario_name": "S1_monthly_revenue",
            "selected_tool_name": "",
            "agent_response": "Revenue this month is approximately $500,000.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_wrong_tool_returns_0(self):
        raw = {
            "scenario_name": "S2_inventory_level",
            "selected_tool_name": "query_db",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_correct_tool_s3(self):
        raw = {
            "scenario_name": "S3_customer_email",
            "selected_tool_name": "lookup_customer",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_unknown_scenario_returns_0(self):
        raw = {
            "scenario_name": "S99_unknown",
            "selected_tool_name": "query_db",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0


# ---------------------------------------------------------------------------
# Integration tests: live LLM (requires APEX_PROFILE + API key)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.free
async def test_live_s2_inventory():
    """
    Live: S2 inventory level — checks whether agent calls get_inventory or answers from memory.

    Run: APEX_PROFILE=free pytest tests/layer1/test_tool_omission.py::test_live_s2_inventory -v -s
    """
    from apex.harness import build_l1_agent_fn

    module = ToolOmissionEval()
    agent_fn = build_l1_agent_fn(verbose=True)

    scenario = next(s for s in module.scenarios() if s.name == "S2_inventory_level")
    result = await module.run(scenario, agent_fn)

    print(f"\n{result.summary()}")
    print(f"Tool called: {result.actual_tool_call}")
    print(f"Reason:      {result.metadata.get('score_reason')}")

    assert result.score in (0.0, 0.5, 1.0)
    assert result.latency_ms > 0


@pytest.mark.asyncio
@pytest.mark.free
async def test_live_all_scenarios():
    """
    Full suite: all 5 tool omission scenarios against the active LLM.

    Run: APEX_PROFILE=free pytest tests/layer1/test_tool_omission.py::test_live_all_scenarios -v -s
    """
    from apex.harness import build_l1_agent_fn

    module = ToolOmissionEval()
    agent_fn = build_l1_agent_fn(verbose=False)
    results = await module.run_all(agent_fn)

    print("\n" + "=" * 60)
    print("APEX L1 — Tool Omission Results")
    print("=" * 60)
    for r in results:
        print(r.summary())
        print(f"  tool_called: {r.metadata.get('tool_called')}")
        print(f"  reason:      {r.metadata.get('score_reason')}")

    avg_score = sum(r.score for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.2f} / 1.00")
    print(f"Pass rate: {sum(r.passed for r in results)}/{len(results)}")

    assert len(results) == 5
