"""
tests/layer2/test_semantic_arg_error.py
----------------------------------------
APEX Layer 2 — Semantic Argument Error
pytest suite: 5 scenarios × scored intent validation

Run (free tier, no API cost for fixture replay):
    APEX_PROFILE=free pytest tests/layer2/test_semantic_arg_error.py -v

Run (standard, live LLM calls):
    APEX_PROFILE=standard pytest tests/layer2/test_semantic_arg_error.py -v -m standard
"""

import pytest
from apex.layer2_input_construction.semantic_arg_error import (
    SemanticArgErrorEval,
    SCENARIOS,
    _validate_date_range,
    _validate_aggregation,
    _validate_filter,
    _validate_metric,
    _validate_timezone,
)
from apex.base import FailureLayer, DetectionDifficulty


# ---------------------------------------------------------------------------
# Unit tests: validators (zero LLM cost — pure logic)
# ---------------------------------------------------------------------------

class TestValidators:
    """Test the intent validators directly. Zero API calls."""

    def test_date_range_rolling_window_fails(self):
        bad_sql = "SELECT SUM(revenue) FROM orders WHERE created_at >= date('now', '-90 days')"
        passed, reason = _validate_date_range(bad_sql, {})
        assert not passed
        assert "90-day" in reason or "rolling" in reason

    def test_date_range_quarter_boundary_passes(self):
        good_sql = "SELECT SUM(revenue) FROM orders WHERE created_at >= '2026-01-01' AND created_at < '2026-04-01'"
        passed, reason = _validate_date_range(good_sql, {})
        assert passed

    def test_aggregation_raw_avg_fails(self):
        bad_sql = "SELECT AVG(revenue) FROM orders WHERE strftime('%Y-%m', created_at) = '2026-04'"
        passed, reason = _validate_aggregation(bad_sql, {})
        assert not passed
        assert "row" in reason or "grain" in reason

    def test_aggregation_daily_grain_passes(self):
        good_sql = """
            SELECT AVG(daily_rev) FROM (
                SELECT date(created_at) AS d, SUM(revenue) AS daily_rev
                FROM orders
                WHERE strftime('%Y-%m', created_at) = '2026-04'
                GROUP BY date(created_at)
            )
        """
        passed, reason = _validate_aggregation(good_sql, {})
        assert passed

    def test_filter_inversion_fails(self):
        bad_sql = "SELECT SUM(revenue) FROM orders WHERE region = 'RETURN'"
        passed, reason = _validate_filter(bad_sql, {})
        assert not passed
        assert "includes" in reason or "instead" in reason

    def test_filter_exclusion_passes(self):
        good_sql = "SELECT SUM(revenue) FROM orders WHERE region != 'RETURN'"
        passed, reason = _validate_filter(good_sql, {})
        assert passed

    def test_metric_revenue_only_fails(self):
        bad_sql = "SELECT region, SUM(revenue) FROM orders GROUP BY region"
        passed, reason = _validate_metric(bad_sql, {})
        assert not passed
        assert "margin" in reason or "cogs" in reason

    def test_metric_margin_calc_passes(self):
        good_sql = """
            SELECT region,
                   (SUM(revenue) - SUM(cogs)) / SUM(revenue) AS margin
            FROM orders GROUP BY region
        """
        passed, reason = _validate_metric(good_sql, {})
        assert passed

    def test_timezone_no_conversion_fails(self):
        bad_sql = "SELECT COUNT(*) FROM orders WHERE date(created_at) = '2026-05-30'"
        passed, reason = _validate_timezone(bad_sql, {})
        assert not passed
        assert "timezone" in reason or "UTC" in reason

    def test_timezone_with_offset_passes(self):
        good_sql = "SELECT COUNT(*) FROM orders WHERE datetime(created_at, '-4 hours') >= '2026-05-30 00:00:00'"
        passed, reason = _validate_timezone(good_sql, {})
        assert passed


# ---------------------------------------------------------------------------
# Module tests: structure & metadata (zero LLM cost)
# ---------------------------------------------------------------------------

class TestEvalModuleStructure:

    def setup_method(self):
        self.module = SemanticArgErrorEval()

    def test_failure_mode_identifier(self):
        assert self.module.failure_mode == "semantic_argument_error"

    def test_layer_assignment(self):
        assert self.module.layer == FailureLayer.L2_INPUT_CONSTRUCTION

    def test_detection_difficulty(self):
        assert self.module.detection_difficulty == DetectionDifficulty.HIGH

    def test_scenarios_count(self):
        assert len(self.module.scenarios()) == 5

    def test_all_scenarios_have_injected_fault(self):
        for s in self.module.scenarios():
            assert s.injected_fault, f"{s.name} missing injected_fault"
            assert "type" in s.injected_fault
            assert "description" in s.injected_fault

    def test_all_scenarios_have_tool_schema(self):
        for s in self.module.scenarios():
            assert "name" in s.tool_schema
            assert "parameters" in s.tool_schema


# ---------------------------------------------------------------------------
# Scorer unit tests (zero LLM cost)
# ---------------------------------------------------------------------------

class TestScorer:

    def setup_method(self):
        self.module = SemanticArgErrorEval()

    def test_score_correct_sql_returns_1(self):
        raw = {
            "scenario_name": "S3_filter_inversion",
            "sql": "SELECT SUM(revenue) FROM orders WHERE region != 'RETURN'",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_wrong_sql_returns_0(self):
        raw = {
            "scenario_name": "S3_filter_inversion",
            "sql": "SELECT SUM(revenue) FROM orders WHERE region = 'RETURN'",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_unknown_scenario_returns_0(self):
        raw = {
            "scenario_name": "S99_unknown",
            "sql": "SELECT 1",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0


# ---------------------------------------------------------------------------
# Integration tests: live LLM (requires APEX_PROFILE + API key)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.free
async def test_live_s3_filter_inversion():
    """
    Live integration: run S3 (filter inversion) against Groq Llama-3.1-8b.
    Expects the model to get it right (or captures the failure for analysis).

    Run: APEX_PROFILE=free pytest tests/layer2/test_semantic_arg_error.py::test_live_s3_filter_inversion -v
    """
    from apex.harness import build_agent_fn

    module = SemanticArgErrorEval()
    agent_fn = build_agent_fn(verbose=True)

    # Run only S3 for speed
    scenario = next(s for s in module.scenarios() if s.name == "S3_filter_inversion")
    result = await module.run(scenario, agent_fn)

    print(f"\n{result.summary()}")
    print(f"SQL: {result.actual_tool_call}")
    print(f"Reason: {result.metadata.get('score_reason')}")

    # We don't assert pass/fail here — we record the result.
    # This is an eval, not a correctness test. The score IS the finding.
    assert result.score in (0.0, 0.5, 1.0)
    assert result.latency_ms > 0


@pytest.mark.asyncio
@pytest.mark.free
async def test_live_all_scenarios():
    """
    Full suite: all 5 L2 semantic error scenarios against Groq.
    Prints a mini report. Does not assert pass — records findings.

    Run: APEX_PROFILE=free pytest tests/layer2/test_semantic_arg_error.py::test_live_all_scenarios -v -s
    """
    from apex.harness import build_agent_fn

    module = SemanticArgErrorEval()
    agent_fn = build_agent_fn(verbose=False)
    results = await module.run_all(agent_fn)

    print("\n" + "=" * 60)
    print("APEX L2 — Semantic Argument Error Results")
    print("=" * 60)
    for r in results:
        print(r.summary())

    avg_score = sum(r.score for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.2f} / 1.00")
    print(f"Pass rate: {sum(r.passed for r in results)}/{len(results)}")

    assert len(results) == 5
