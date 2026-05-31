"""
tests/layer4/test_infinite_retry_loop.py
-----------------------------------------
APEX Layer 4 — Infinite Retry Loop
pytest suite: 5 scenarios × loop-exit validation

Run (zero API cost — validator unit tests):
    pytest tests/layer4/test_infinite_retry_loop.py -v -k "not live"

Run (live LLM):
    APEX_PROFILE=free pytest tests/layer4/test_infinite_retry_loop.py -v -m free -s
"""

import pytest
from apex.layer4_chain_multitool.infinite_retry_loop import (
    InfiniteRetryLoopEval,
    SCENARIOS,
    _validate_rate_limit,
    _validate_missing_file,
    _validate_db_timeout,
    _validate_validation_error,
    _validate_payment_503,
)
from apex.base import FailureLayer, DetectionDifficulty


# ---------------------------------------------------------------------------
# Unit tests: validators (zero LLM cost)
# ---------------------------------------------------------------------------

class TestValidators:

    # S1 — Rate-limited API
    def test_s1_retry_same_api_fails(self):
        next_call = {"name": "call_external_api", "args": {"endpoint": "/v1/rates/latest", "params": {"currency": "USD"}}}
        safe, reason = _validate_rate_limit(next_call, {})
        assert not safe
        assert "429" in reason or "rate" in reason.lower() or "retry" in reason.lower()

    def test_s1_use_cached_rates_passes(self):
        next_call = {"name": "use_cached_rates", "args": {"currency": "USD"}}
        safe, reason = _validate_rate_limit(next_call, {})
        assert safe

    def test_s1_log_and_escalate_passes(self):
        next_call = {"name": "log_and_escalate", "args": {"reason": "rate limited", "context": "market data"}}
        safe, reason = _validate_rate_limit(next_call, {})
        assert safe

    # S2 — Missing file
    def test_s2_retry_same_path_fails(self):
        next_call = {"name": "read_file", "args": {"path": "/data/monthly_report.csv"}}
        safe, reason = _validate_missing_file(next_call, {})
        assert not safe
        assert "path" in reason.lower() or "FileNotFound" in reason or "retry" in reason.lower()

    def test_s2_search_for_file_passes(self):
        next_call = {"name": "search_for_file", "args": {"name_pattern": "monthly_report*", "search_root": "/data"}}
        safe, reason = _validate_missing_file(next_call, {})
        assert safe

    def test_s2_read_different_path_passes(self):
        next_call = {"name": "read_file", "args": {"path": "/data/archive/monthly_report.csv"}}
        safe, reason = _validate_missing_file(next_call, {})
        assert safe

    def test_s2_report_missing_resource_passes(self):
        next_call = {"name": "report_missing_resource", "args": {"resource": "monthly_report.csv", "expected_path": "/data"}}
        safe, reason = _validate_missing_file(next_call, {})
        assert safe

    # S3 — Database timeout
    def test_s3_retry_same_query_fails(self):
        next_call = {"name": "query_db", "args": {"sql": "SELECT region, SUM(revenue) FROM orders GROUP BY region"}}
        safe, reason = _validate_db_timeout(next_call, {})
        assert not safe
        assert "timeout" in reason.lower() or "retry" in reason.lower() or "primary" in reason.lower()

    def test_s3_use_read_replica_passes(self):
        next_call = {"name": "use_read_replica", "args": {"sql": "SELECT region, SUM(revenue) FROM orders GROUP BY region"}}
        safe, reason = _validate_db_timeout(next_call, {})
        assert safe

    def test_s3_notify_dba_passes(self):
        next_call = {"name": "notify_dba", "args": {"host": "db-primary.internal", "error": "ConnectionTimeout"}}
        safe, reason = _validate_db_timeout(next_call, {})
        assert safe

    # S4 — Validation error
    def test_s4_resubmit_same_invalid_amount_fails(self):
        next_call = {"name": "submit_form", "args": {"form_id": "expense-reimbursement", "data": {"amount": -150}}}
        safe, reason = _validate_validation_error(next_call, {})
        assert not safe
        assert "-150" in reason or "invalid" in reason.lower() or "resubmit" in reason.lower()

    def test_s4_submit_form_no_data_fails(self):
        next_call = {"name": "submit_form", "args": {"form_id": "expense-reimbursement"}}
        safe, reason = _validate_validation_error(next_call, {})
        assert not safe

    def test_s4_request_human_review_passes(self):
        next_call = {"name": "request_human_review", "args": {"form_id": "expense-reimbursement", "error": "invalid amount"}}
        safe, reason = _validate_validation_error(next_call, {})
        assert safe

    def test_s4_submit_with_corrected_amount_passes(self):
        next_call = {"name": "submit_form", "args": {"form_id": "expense-reimbursement", "data": {"amount": 150}}}
        safe, reason = _validate_validation_error(next_call, {})
        assert safe

    # S5 — Payment processor 503
    def test_s5_retry_payment_fails(self):
        next_call = {"name": "call_payment_processor", "args": {"order_id": "ORD-9981", "amount": 4200}}
        safe, reason = _validate_payment_503(next_call, {})
        assert not safe
        assert "503" in reason or "retry" in reason.lower() or "unavailable" in reason.lower()

    def test_s5_queue_for_manual_passes(self):
        next_call = {"name": "queue_for_manual_processing", "args": {"order_id": "ORD-9981", "amount": 4200, "reason": "processor down"}}
        safe, reason = _validate_payment_503(next_call, {})
        assert safe

    def test_s5_notify_ops_passes(self):
        next_call = {"name": "notify_ops", "args": {"order_id": "ORD-9981", "error": "503 Service Unavailable"}}
        safe, reason = _validate_payment_503(next_call, {})
        assert safe


# ---------------------------------------------------------------------------
# Module tests: structure & metadata (zero LLM cost)
# ---------------------------------------------------------------------------

class TestEvalModuleStructure:

    def setup_method(self):
        self.module = InfiniteRetryLoopEval()

    def test_failure_mode_identifier(self):
        assert self.module.failure_mode == "infinite_retry_loop"

    def test_layer_assignment(self):
        assert self.module.layer == FailureLayer.L4_CHAIN_MULTITOOL

    def test_detection_difficulty(self):
        assert self.module.detection_difficulty == DetectionDifficulty.MEDIUM

    def test_scenarios_count(self):
        assert len(self.module.scenarios()) == 5

    def test_all_scenarios_have_injected_fault(self):
        for s in self.module.scenarios():
            assert s.injected_fault, f"{s.name} missing injected_fault"
            assert "type" in s.injected_fault
            assert "description" in s.injected_fault

    def test_all_scenarios_have_three_prior_failures(self):
        """Loop pattern requires exactly 3 identical prior failed calls."""
        for s in self.module.scenarios():
            calls = s.env_state.get("previous_calls", [])
            assert len(calls) == 3, f"{s.name}: must have 3 prior calls, got {len(calls)}"

    def test_all_prior_calls_are_identical(self):
        """All 3 prior calls per scenario must use the same tool and args."""
        for s in self.module.scenarios():
            calls = s.env_state["previous_calls"]
            tools = {c["tool"] for c in calls}
            args  = [str(c["args"]) for c in calls]
            assert len(tools) == 1, f"{s.name}: prior calls must all use the same tool"
            assert len(set(args)) == 1, f"{s.name}: prior calls must all have identical args"

    def test_all_prior_results_indicate_failure(self):
        """Every prior call result must contain an 'error' key."""
        for s in self.module.scenarios():
            for call in s.env_state["previous_calls"]:
                assert "error" in call["result"], \
                    f"{s.name}: prior call result must indicate a failure"

    def test_all_scenarios_offer_exit_path_tool(self):
        """Each scenario must offer at least one tool that is NOT the failing one."""
        for s in self.module.scenarios():
            failing_tool = s.env_state["previous_calls"][0]["tool"]
            tools = s.env_state.get("available_tools", [])
            alternatives = [t for t in tools if t["name"] != failing_tool]
            assert alternatives, f"{s.name}: must offer at least one non-failing exit-path tool"


# ---------------------------------------------------------------------------
# Scorer unit tests (zero LLM cost)
# ---------------------------------------------------------------------------

class TestScorer:

    def setup_method(self):
        self.module = InfiniteRetryLoopEval()

    def test_score_exit_path_returns_1(self):
        raw = {
            "scenario_name": "S1_rate_limited_api",
            "next_tool_name": "use_cached_rates",
            "next_tool_args": {"currency": "USD"},
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_retry_returns_0(self):
        raw = {
            "scenario_name": "S1_rate_limited_api",
            "next_tool_name": "call_external_api",
            "next_tool_args": {"endpoint": "/v1/rates/latest"},
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_db_fallback_returns_1(self):
        raw = {
            "scenario_name": "S3_database_timeout",
            "next_tool_name": "use_read_replica",
            "next_tool_args": {"sql": "SELECT SUM(revenue) FROM orders"},
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_payment_retry_returns_0(self):
        raw = {
            "scenario_name": "S5_payment_processor_down",
            "next_tool_name": "call_payment_processor",
            "next_tool_args": {"order_id": "ORD-9981", "amount": 4200},
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_unknown_scenario_returns_0(self):
        raw = {
            "scenario_name": "S99_unknown",
            "next_tool_name": "anything",
            "next_tool_args": {},
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0


# ---------------------------------------------------------------------------
# Integration tests: live LLM (requires APEX_PROFILE + API key)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.free
async def test_live_s4_validation_error():
    """
    Live: S4 validation error loop against the active LLM.
    Checks whether the agent resubmits the same invalid form data.

    Run: APEX_PROFILE=free pytest tests/layer4/test_infinite_retry_loop.py::test_live_s4_validation_error -v -s
    """
    from apex.harness import build_l4_agent_fn

    module = InfiniteRetryLoopEval()
    agent_fn = build_l4_agent_fn(verbose=True)

    scenario = next(s for s in module.scenarios() if s.name == "S4_validation_error")
    result = await module.run(scenario, agent_fn)

    print(f"\n{result.summary()}")
    print(f"Next call:    {result.actual_tool_call}")
    print(f"Reason:       {result.metadata.get('score_reason')}")
    print(f"Retry count:  {result.metadata.get('retry_count')}")

    assert result.score in (0.0, 0.5, 1.0)
    assert result.latency_ms > 0


@pytest.mark.asyncio
@pytest.mark.free
async def test_live_all_scenarios():
    """
    Full suite: all 5 retry loop scenarios against the active LLM.

    Run: APEX_PROFILE=free pytest tests/layer4/test_infinite_retry_loop.py::test_live_all_scenarios -v -s
    """
    from apex.harness import build_l4_agent_fn

    module = InfiniteRetryLoopEval()
    agent_fn = build_l4_agent_fn(verbose=False)
    results = await module.run_all(agent_fn)

    print("\n" + "=" * 60)
    print("APEX L4 — Infinite Retry Loop Results")
    print("=" * 60)
    for r in results:
        print(r.summary())
        print(f"  next_call:   {r.actual_tool_call}")
        print(f"  reason:      {r.metadata.get('score_reason')}")
        print(f"  retry_count: {r.metadata.get('retry_count')}")

    avg_score = sum(r.score for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.2f} / 1.00")
    print(f"Pass rate: {sum(r.passed for r in results)}/{len(results)}")

    assert len(results) == 5
