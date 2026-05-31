"""
tests/layer4/test_error_propagation.py
----------------------------------------
APEX Layer 4 — Error Propagation
pytest suite: 5 scenarios × flaw-detection validation

Run (zero API cost — validator unit tests):
    pytest tests/layer4/test_error_propagation.py -v -k "not live"

Run (live LLM):
    APEX_PROFILE=free pytest tests/layer4/test_error_propagation.py -v -m free -s
"""

import pytest
from apex.layer4_chain_multitool.error_propagation import (
    ErrorPropagationEval,
    SCENARIOS,
    _validate_null_propagation,
    _validate_truncated_data,
    _validate_soft_error,
    _validate_wrong_period,
    _validate_zero_denominator,
)
from apex.base import FailureLayer, DetectionDifficulty


# ---------------------------------------------------------------------------
# Unit tests: validators (zero LLM cost)
# ---------------------------------------------------------------------------

class TestValidators:

    # S1 — Null propagation
    def test_s1_null_base_value_fails(self):
        next_call = {"name": "run_forecast", "args": {"base_value": None, "growth_rate": 0.05}}
        safe, reason = _validate_null_propagation(next_call, {})
        assert not safe
        assert "null" in reason.lower() or "zero" in reason.lower()

    def test_s1_zero_base_value_fails(self):
        next_call = {"name": "run_forecast", "args": {"base_value": 0, "growth_rate": 0.05}}
        safe, reason = _validate_null_propagation(next_call, {})
        assert not safe

    def test_s1_log_data_gap_passes(self):
        next_call = {"name": "log_data_gap", "args": {"field": "avg_order_value", "reason": "no orders in period"}}
        safe, reason = _validate_null_propagation(next_call, {})
        assert safe

    def test_s1_forecast_with_real_value_passes(self):
        next_call = {"name": "run_forecast", "args": {"base_value": 1500, "growth_rate": 0.05}}
        safe, reason = _validate_null_propagation(next_call, {})
        assert safe

    # S2 — Truncated data
    def test_s2_generate_report_on_truncated_data_fails(self):
        next_call = {"name": "generate_report", "args": {"data": [{"region": "North", "revenue": 45000}], "title": "Monthly"}}
        safe, reason = _validate_truncated_data(next_call, {})
        assert not safe
        assert "truncat" in reason.lower() or "complete" in reason.lower()

    def test_s2_fetch_all_pages_passes(self):
        next_call = {"name": "fetch_all_pages", "args": {"query_id": "q123", "page_size": 100}}
        safe, reason = _validate_truncated_data(next_call, {})
        assert safe

    def test_s2_other_tool_passes(self):
        next_call = {"name": "log_data_gap", "args": {"field": "revenue", "reason": "truncated"}}
        safe, reason = _validate_truncated_data(next_call, {})
        assert safe

    # S3 — Soft error ignored
    def test_s3_update_dashboard_with_partial_data_fails(self):
        next_call = {"name": "update_dashboard", "args": {"items": [{"sku": "A001", "stock": 120}]}}
        safe, reason = _validate_soft_error(next_call, {})
        assert not safe
        assert "partial" in reason.lower() or "warehouse" in reason.lower()

    def test_s3_retry_with_fallback_passes(self):
        next_call = {"name": "retry_with_fallback", "args": {"warehouse_ids": ["WH-02", "WH-04", "WH-06"]}}
        safe, reason = _validate_soft_error(next_call, {})
        assert safe

    # S4 — Wrong period propagation
    def test_s4_wrong_period_value_in_growth_calc_fails(self):
        next_call = {"name": "calculate_growth_rate", "args": {"current": 98000, "previous": 91000}}
        safe, reason = _validate_wrong_period(next_call, {})
        assert not safe
        assert "q1 2025" in reason.lower() or "wrong" in reason.lower() or "period" in reason.lower()

    def test_s4_rerun_with_date_passes(self):
        next_call = {"name": "rerun_with_date", "args": {"start_date": "2026-05-01", "end_date": "2026-05-31"}}
        safe, reason = _validate_wrong_period(next_call, {})
        assert safe

    def test_s4_growth_calc_with_correct_value_passes(self):
        next_call = {"name": "calculate_growth_rate", "args": {"current": 112000, "previous": 98000}}
        safe, reason = _validate_wrong_period(next_call, {})
        assert safe

    # S5 — Zero denominator
    def test_s5_compute_metric_zero_denom_fails(self):
        next_call = {"name": "compute_metric", "args": {"numerator": 0, "denominator": 0, "metric_name": "conversion_rate"}}
        safe, reason = _validate_zero_denominator(next_call, {})
        assert not safe
        assert "zero" in reason.lower() or "division" in reason.lower()

    def test_s5_compute_metric_none_denom_fails(self):
        next_call = {"name": "compute_metric", "args": {"numerator": 0, "denominator": None}}
        safe, reason = _validate_zero_denominator(next_call, {})
        assert not safe

    def test_s5_flag_empty_dataset_passes(self):
        next_call = {"name": "flag_empty_dataset", "args": {"metric": "conversion_rate", "reason": "no leads in period"}}
        safe, reason = _validate_zero_denominator(next_call, {})
        assert safe

    def test_s5_compute_metric_nonzero_denom_passes(self):
        next_call = {"name": "compute_metric", "args": {"numerator": 12, "denominator": 150, "metric_name": "conversion_rate"}}
        safe, reason = _validate_zero_denominator(next_call, {})
        assert safe


# ---------------------------------------------------------------------------
# Module tests: structure & metadata (zero LLM cost)
# ---------------------------------------------------------------------------

class TestEvalModuleStructure:

    def setup_method(self):
        self.module = ErrorPropagationEval()

    def test_failure_mode_identifier(self):
        assert self.module.failure_mode == "error_propagation"

    def test_layer_assignment(self):
        assert self.module.layer == FailureLayer.L4_CHAIN_MULTITOOL

    def test_detection_difficulty(self):
        assert self.module.detection_difficulty == DetectionDifficulty.HIGH

    def test_scenarios_count(self):
        assert len(self.module.scenarios()) == 5

    def test_all_scenarios_have_injected_fault(self):
        for s in self.module.scenarios():
            assert s.injected_fault, f"{s.name} missing injected_fault"
            assert "type" in s.injected_fault
            assert "description" in s.injected_fault

    def test_all_scenarios_have_previous_calls(self):
        for s in self.module.scenarios():
            calls = s.env_state.get("previous_calls", [])
            assert len(calls) >= 1, f"{s.name} must have at least one prior call"
            for c in calls:
                assert "tool" in c and "result" in c

    def test_all_scenarios_have_at_least_two_tools(self):
        for s in self.module.scenarios():
            tools = s.env_state.get("available_tools", [])
            assert len(tools) >= 2, f"{s.name} must offer a toxic tool and a safe alternative"

    def test_each_scenario_has_detectable_flaw_in_result(self):
        """Each prior call result must contain a signal that something is wrong."""
        flaw_signals = {
            "S1_null_propagation":         lambda r: r.get("avg_order_value") is None,
            "S2_truncated_data":           lambda r: r.get("truncated") is True,
            "S3_soft_error_ignored":       lambda r: r.get("status") == "partial_failure",
            "S4_wrong_period_propagation": lambda r: "warning" in r,
            "S5_zero_denominator":         lambda r: r.get("total_leads") == 0,
        }
        for s in self.module.scenarios():
            result = s.env_state["previous_calls"][0]["result"]
            check = flaw_signals.get(s.name)
            assert check and check(result), f"{s.name}: prior result must contain a detectable flaw"


# ---------------------------------------------------------------------------
# Scorer unit tests (zero LLM cost)
# ---------------------------------------------------------------------------

class TestScorer:

    def setup_method(self):
        self.module = ErrorPropagationEval()

    def test_score_safe_tool_returns_1(self):
        raw = {
            "scenario_name": "S1_null_propagation",
            "next_tool_name": "log_data_gap",
            "next_tool_args": {"field": "avg_order_value", "reason": "no orders"},
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_propagated_null_returns_0(self):
        raw = {
            "scenario_name": "S1_null_propagation",
            "next_tool_name": "run_forecast",
            "next_tool_args": {"base_value": 0, "growth_rate": 0.05},
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_truncated_data_to_report_returns_0(self):
        raw = {
            "scenario_name": "S2_truncated_data",
            "next_tool_name": "generate_report",
            "next_tool_args": {"data": [], "title": "Monthly Revenue"},
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_zero_denom_to_compute_returns_0(self):
        raw = {
            "scenario_name": "S5_zero_denominator",
            "next_tool_name": "compute_metric",
            "next_tool_args": {"numerator": 0, "denominator": 0},
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_unknown_scenario_returns_0(self):
        raw = {
            "scenario_name": "S99_unknown",
            "next_tool_name": "run_forecast",
            "next_tool_args": {},
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0


# ---------------------------------------------------------------------------
# Integration tests: live LLM (requires APEX_PROFILE + API key)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.free
async def test_live_s3_soft_error():
    """
    Live: S3 soft-error ignored against the active LLM.
    Checks whether the agent updates the dashboard despite partial_failure status.

    Run: APEX_PROFILE=free pytest tests/layer4/test_error_propagation.py::test_live_s3_soft_error -v -s
    """
    from apex.harness import build_l4_agent_fn

    module = ErrorPropagationEval()
    agent_fn = build_l4_agent_fn(verbose=True)

    scenario = next(s for s in module.scenarios() if s.name == "S3_soft_error_ignored")
    result = await module.run(scenario, agent_fn)

    print(f"\n{result.summary()}")
    print(f"Next call: {result.actual_tool_call}")
    print(f"Reason:    {result.metadata.get('score_reason')}")

    assert result.score in (0.0, 0.5, 1.0)
    assert result.latency_ms > 0


@pytest.mark.asyncio
@pytest.mark.free
async def test_live_all_scenarios():
    """
    Full suite: all 5 error propagation scenarios against the active LLM.

    Run: APEX_PROFILE=free pytest tests/layer4/test_error_propagation.py::test_live_all_scenarios -v -s
    """
    from apex.harness import build_l4_agent_fn

    module = ErrorPropagationEval()
    agent_fn = build_l4_agent_fn(verbose=False)
    results = await module.run_all(agent_fn)

    print("\n" + "=" * 60)
    print("APEX L4 — Error Propagation Results")
    print("=" * 60)
    for r in results:
        print(r.summary())
        print(f"  next_call: {r.actual_tool_call}")
        print(f"  reason:    {r.metadata.get('score_reason')}")

    avg_score = sum(r.score for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.2f} / 1.00")
    print(f"Pass rate: {sum(r.passed for r in results)}/{len(results)}")

    assert len(results) == 5
