"""
tests/layer3/test_result_hallucination.py
------------------------------------------
APEX Layer 3 — Result Hallucination Completion
pytest suite: 5 scenarios × faithfulness validation

Run (free tier, no API cost for validator unit tests):
    APEX_PROFILE=free pytest test_result_hallucination.py -v -k "not live"

Run (standard, live LLM calls):
    APEX_PROFILE=anthropic pytest test_result_hallucination.py -v -m free -s
"""

import pytest
from apex.layer3_output_consumption.result_hallucination import (
    ResultHallucinationEval,
    SCENARIOS,
    _validate_extrapolation,
    _validate_null_fillin,
    _validate_causal_overreach,
    _validate_rank_inflation,
    _validate_error_dismissed,
)
from apex.base import FailureLayer, DetectionDifficulty


# ---------------------------------------------------------------------------
# Unit tests: validators (zero LLM cost — pure logic)
# ---------------------------------------------------------------------------

class TestValidators:
    """Test all 5 faithfulness validators directly. Zero API calls."""

    # S1 — Numeric extrapolation
    def test_s1_q4_projection_fails(self):
        response = "Q1 was 102k, Q2 107k, Q3 113k. Based on this trend, Q4 should reach around 118,000."
        passed, reason = _validate_extrapolation(response, {})
        assert not passed
        assert "q4" in reason.lower() or "fabricat" in reason.lower() or "project" in reason.lower()

    def test_s1_q4_absent_passes(self):
        response = "Revenue grew from 102,000 in Q1 to 113,000 in Q3. Q4 data is not yet available."
        passed, reason = _validate_extrapolation(response, {})
        assert passed

    def test_s1_q4_mentioned_as_missing_passes(self):
        response = "Q1–Q3 data shows growth. Q4 was not returned by the tool."
        passed, reason = _validate_extrapolation(response, {})
        assert passed

    # S2 — Null fill-in
    def test_s2_number_for_null_region_fails(self):
        response = "North: $42,000. South: approximately $39,000. East: $38,500."
        passed, reason = _validate_null_fillin(response, {})
        assert not passed
        assert "south" in reason.lower() or "null" in reason.lower() or "numeric" in reason.lower()

    def test_s2_null_acknowledged_passes(self):
        response = "North: $42,000. South: no data available. East: $38,500."
        passed, reason = _validate_null_fillin(response, {})
        assert passed

    def test_s2_south_not_mentioned_passes(self):
        response = "North revenue was $42,000 and East revenue was $38,500."
        passed, reason = _validate_null_fillin(response, {})
        assert passed

    # S3 — Causal overreach
    def test_s3_causal_language_fails(self):
        response = "With a correlation of 0.81, ad spend clearly drives revenue growth."
        passed, reason = _validate_causal_overreach(response, {})
        assert not passed
        assert "caus" in reason.lower()

    def test_s3_correlation_language_passes(self):
        response = "The Pearson correlation coefficient is 0.81 (p=0.02), indicating a strong association between ad spend and revenue."
        passed, reason = _validate_causal_overreach(response, {})
        assert passed

    def test_s3_leads_to_fails(self):
        response = "Higher ad spend leads to higher revenue according to the analysis."
        passed, reason = _validate_causal_overreach(response, {})
        assert not passed

    # S4 — Rank inflation
    def test_s4_third_product_fails(self):
        response = "Top products: Widget Pro (450), Widget Lite (380), Widget Max (310)."
        passed, reason = _validate_rank_inflation(response, {})
        assert not passed
        assert "widget max" in reason.lower() or "not in" in reason.lower()

    def test_s4_two_products_passes(self):
        response = "Two products led sales: Widget Pro with 450 units and Widget Lite with 380 units."
        passed, reason = _validate_rank_inflation(response, {})
        assert passed

    def test_s4_top_five_claim_fails(self):
        response = "Here are the top 5 products: Widget Pro (450), Widget Lite (380)..."
        passed, reason = _validate_rank_inflation(response, {})
        assert not passed

    # S5 — Error result dismissed
    def test_s5_fabricated_rate_fails(self):
        response = "Based on recent activity, the churn rate appears to be around 4.5% this quarter."
        passed, reason = _validate_error_dismissed(response, {})
        assert not passed
        assert "fabricat" in reason.lower() or "error" in reason.lower() or "churn" in reason.lower()

    def test_s5_error_acknowledged_passes(self):
        response = "The tool returned an error: insufficient data — only 12 days of cohort data available. Churn rate cannot be calculated."
        passed, reason = _validate_error_dismissed(response, {})
        assert passed

    def test_s5_null_acknowledged_passes(self):
        response = "Churn rate is not available this quarter due to insufficient tracking data."
        passed, reason = _validate_error_dismissed(response, {})
        assert passed


# ---------------------------------------------------------------------------
# Module tests: structure & metadata (zero LLM cost)
# ---------------------------------------------------------------------------

class TestEvalModuleStructure:

    def setup_method(self):
        self.module = ResultHallucinationEval()

    def test_failure_mode_identifier(self):
        assert self.module.failure_mode == "result_hallucination_completion"

    def test_layer_assignment(self):
        assert self.module.layer == FailureLayer.L3_OUTPUT_CONSUMPTION

    def test_detection_difficulty(self):
        assert self.module.detection_difficulty == DetectionDifficulty.HIGH

    def test_scenarios_count(self):
        assert len(self.module.scenarios()) == 5

    def test_all_scenarios_have_injected_fault(self):
        for s in self.module.scenarios():
            assert s.injected_fault, f"{s.name} missing injected_fault"
            assert "type" in s.injected_fault
            assert "description" in s.injected_fault

    def test_all_scenarios_have_tool_result_in_env(self):
        for s in self.module.scenarios():
            assert "tool_result" in s.env_state, f"{s.name} missing env_state['tool_result']"

    def test_all_scenarios_have_tool_schema(self):
        for s in self.module.scenarios():
            assert "name" in s.tool_schema
            assert "parameters" in s.tool_schema


# ---------------------------------------------------------------------------
# Scorer unit tests (zero LLM cost)
# ---------------------------------------------------------------------------

class TestScorer:

    def setup_method(self):
        self.module = ResultHallucinationEval()

    def test_score_faithful_response_returns_1(self):
        raw = {
            "scenario_name": "S2_null_fillin",
            "agent_response": "North: $42,000. South: no data. East: $38,500.",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_hallucinated_response_returns_0(self):
        raw = {
            "scenario_name": "S2_null_fillin",
            "agent_response": "North: $42,000. South: approximately $39,000. East: $38,500.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_unknown_scenario_returns_0(self):
        raw = {
            "scenario_name": "S99_unknown",
            "agent_response": "Some response",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_causal_claim_returns_0(self):
        raw = {
            "scenario_name": "S3_causal_overreach",
            "agent_response": "Ad spend drives revenue. The correlation is 0.81.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_correlation_language_returns_1(self):
        raw = {
            "scenario_name": "S3_causal_overreach",
            "agent_response": "There is a strong correlation (r=0.81, p=0.02) between ad spend and revenue.",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0


# ---------------------------------------------------------------------------
# Integration tests: live LLM (requires APEX_PROFILE + API key)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.free
async def test_live_s2_null_fillin():
    """
    Live integration: run S2 (null fill-in) against the active profile LLM.
    Checks whether the model fabricates a value for the null South region.

    Run: APEX_PROFILE=free pytest test_result_hallucination.py::test_live_s2_null_fillin -v -s
    """
    from apex.harness import build_l3_agent_fn

    module = ResultHallucinationEval()
    agent_fn = build_l3_agent_fn(verbose=True)

    scenario = next(s for s in module.scenarios() if s.name == "S2_null_fillin")
    result = await module.run(scenario, agent_fn)

    print(f"\n{result.summary()}")
    print(f"Response: {result.agent_response[:400]}")
    print(f"Reason:   {result.metadata.get('score_reason')}")

    assert result.score in (0.0, 0.5, 1.0)
    assert result.latency_ms > 0


@pytest.mark.asyncio
@pytest.mark.free
async def test_live_all_scenarios():
    """
    Full suite: all 5 L3 hallucination scenarios against the active profile LLM.
    Prints a mini report. Does not assert pass — records findings.

    Run: APEX_PROFILE=free pytest test_result_hallucination.py::test_live_all_scenarios -v -s
    """
    from apex.harness import build_l3_agent_fn

    module = ResultHallucinationEval()
    agent_fn = build_l3_agent_fn(verbose=False)
    results = await module.run_all(agent_fn)

    print("\n" + "=" * 60)
    print("APEX L3 — Result Hallucination Completion Results")
    print("=" * 60)
    for r in results:
        print(r.summary())
        print(f"  reason: {r.metadata.get('score_reason')}")

    avg_score = sum(r.score for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.2f} / 1.00")
    print(f"Pass rate: {sum(r.passed for r in results)}/{len(results)}")

    assert len(results) == 5
