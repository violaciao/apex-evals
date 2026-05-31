"""
tests/layer1/test_false_tool_trigger.py
----------------------------------------
APEX Layer 1 — False Tool Trigger
pytest suite: 5 scenarios × unnecessary-tool-call detection

Run (zero API cost — validator unit tests):
    pytest tests/layer1/test_false_tool_trigger.py -v -k "not live"

Run (live LLM):
    APEX_PROFILE=free pytest tests/layer1/test_false_tool_trigger.py -v -m free -s
"""

import pytest
from apex.layer1_tool_selection.false_tool_trigger import (
    FalseToolTriggerEval,
    SCENARIOS,
    _no_tool_expected,
)
from apex.base import FailureLayer, DetectionDifficulty


# ---------------------------------------------------------------------------
# Unit tests: validator (zero LLM cost)
# ---------------------------------------------------------------------------

class TestValidator:

    def test_no_tool_called_passes(self):
        score, reason = _no_tool_expected("", "SaaS stands for Software as a Service.", {})
        assert score == 1.0
        assert "without calling" in reason.lower()

    def test_tool_called_fails(self):
        score, reason = _no_tool_expected("search_knowledge_base", "", {})
        assert score == 0.0
        assert "search_knowledge_base" in reason

    def test_any_tool_called_fails(self):
        score, reason = _no_tool_expected("query_db", "SELECT ...", {})
        assert score == 0.0
        assert "unnecessarily" in reason.lower()


# ---------------------------------------------------------------------------
# Module tests: structure & metadata (zero LLM cost)
# ---------------------------------------------------------------------------

class TestEvalModuleStructure:

    def setup_method(self):
        self.module = FalseToolTriggerEval()

    def test_failure_mode_identifier(self):
        assert self.module.failure_mode == "false_tool_trigger"

    def test_layer_assignment(self):
        assert self.module.layer == FailureLayer.L1_TOOL_SELECTION

    def test_detection_difficulty(self):
        assert self.module.detection_difficulty == DetectionDifficulty.MEDIUM

    def test_scenarios_count(self):
        assert len(self.module.scenarios()) == 5

    def test_all_scenarios_have_injected_fault(self):
        for s in self.module.scenarios():
            assert s.injected_fault, f"{s.name} missing injected_fault"
            assert "type" in s.injected_fault
            assert "description" in s.injected_fault

    def test_all_scenarios_expected_tool_is_none(self):
        """Every scenario's expected_tool_call should indicate no tool is needed."""
        for s in self.module.scenarios():
            assert s.expected_tool_call.get("name") is None, (
                f"{s.name}: expected_tool_call.name should be None — no tool should be called"
            )

    def test_all_scenarios_have_available_tools(self):
        """Each scenario provides at least one tool to tempt the agent."""
        for s in self.module.scenarios():
            tools = s.env_state.get("available_tools", [])
            assert len(tools) >= 1, f"{s.name}: must have at least one tempting tool"

    def test_injected_fault_type_is_false_tool_trigger(self):
        for s in self.module.scenarios():
            assert s.injected_fault["type"] == "false_tool_trigger"


# ---------------------------------------------------------------------------
# Scorer unit tests (zero LLM cost)
# ---------------------------------------------------------------------------

class TestScorer:

    def setup_method(self):
        self.module = FalseToolTriggerEval()

    def test_score_no_tool_called_returns_1(self):
        raw = {
            "scenario_name": "S1_acronym_definition",
            "selected_tool_name": "",
            "agent_response": "SaaS stands for Software as a Service.",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_tool_called_returns_0(self):
        raw = {
            "scenario_name": "S1_acronym_definition",
            "selected_tool_name": "search_knowledge_base",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_s2_no_tool_returns_1(self):
        raw = {
            "scenario_name": "S2_unit_conversion",
            "selected_tool_name": "",
            "agent_response": "98.6°F equals 37°C.",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_s3_query_db_returns_0(self):
        raw = {
            "scenario_name": "S3_calendar_fact",
            "selected_tool_name": "query_db",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_unknown_scenario_returns_0(self):
        raw = {
            "scenario_name": "S99_unknown",
            "selected_tool_name": "",
            "agent_response": "",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0


# ---------------------------------------------------------------------------
# Integration tests: live LLM (requires APEX_PROFILE + API key)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.free
async def test_live_s1_acronym():
    """
    Live: S1 acronym definition — checks whether agent triggers a tool unnecessarily.

    Run: APEX_PROFILE=free pytest tests/layer1/test_false_tool_trigger.py::test_live_s1_acronym -v -s
    """
    from apex.harness import build_l1_agent_fn

    module = FalseToolTriggerEval()
    agent_fn = build_l1_agent_fn(verbose=True)

    scenario = next(s for s in module.scenarios() if s.name == "S1_acronym_definition")
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
    Full suite: all 5 false-trigger scenarios against the active LLM.

    Run: APEX_PROFILE=free pytest tests/layer1/test_false_tool_trigger.py::test_live_all_scenarios -v -s
    """
    from apex.harness import build_l1_agent_fn

    module = FalseToolTriggerEval()
    agent_fn = build_l1_agent_fn(verbose=False)
    results = await module.run_all(agent_fn)

    print("\n" + "=" * 60)
    print("APEX L1 — False Tool Trigger Results")
    print("=" * 60)
    for r in results:
        print(r.summary())
        print(f"  tool_called: {r.metadata.get('tool_called')}")
        print(f"  reason:      {r.metadata.get('score_reason')}")

    avg_score = sum(r.score for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.2f} / 1.00")
    print(f"Pass rate: {sum(r.passed for r in results)}/{len(results)}")

    assert len(results) == 5
