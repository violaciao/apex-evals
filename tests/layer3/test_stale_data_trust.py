"""
tests/layer3/test_stale_data_trust.py
--------------------------------------
APEX Layer 3 — Stale Data Trust
pytest suite: 5 scenarios × staleness-acknowledgment validation

Run (zero API cost — validator unit tests):
    pytest tests/layer3/test_stale_data_trust.py -v -k "not live"

Run (live LLM):
    APEX_PROFILE=free pytest tests/layer3/test_stale_data_trust.py -v -m free -s
"""

import pytest
from apex.layer3_output_consumption.stale_data_trust import (
    StaleDataTrustEval,
    SCENARIOS,
    _validate_cached_price,
    _validate_analytics_lag,
    _validate_replica_lag,
    _validate_stale_profile,
    _validate_batch_inventory,
)
from apex.base import FailureLayer, DetectionDifficulty


# ---------------------------------------------------------------------------
# Unit tests: validators (zero LLM cost)
# ---------------------------------------------------------------------------

class TestValidators:

    # S1 — Cached price feed
    def test_s1_mentions_cache_passes(self):
        passed, reason = _validate_cached_price(
            "The price is $149.99, but note this is cached data from 47 minutes ago.", {}
        )
        assert passed
        assert "47" in reason or "cache" in reason.lower()

    def test_s1_presents_as_current_fails(self):
        passed, reason = _validate_cached_price(
            "The current price for SKU-X200 is $149.99.", {}
        )
        assert not passed
        assert "cache" in reason.lower() or "47" in reason

    def test_s1_mentions_cached_keyword_passes(self):
        passed, _ = _validate_cached_price(
            "The cached price shows $149.99, but the live feed is down.", {}
        )
        assert passed

    def test_s1_no_caveat_fails(self):
        passed, _ = _validate_cached_price("SKU-X200 costs $149.99.", {})
        assert not passed

    # S2 — Analytics pipeline lag
    def test_s2_mentions_lag_passes(self):
        passed, _ = _validate_analytics_lag(
            "We had 1,847 active users as of yesterday (18-hour lag in the analytics pipeline).", {}
        )
        assert passed

    def test_s2_presents_as_live_fails(self):
        passed, reason = _validate_analytics_lag(
            "You currently have 1,847 active users.", {}
        )
        assert not passed
        assert "lag" in reason.lower() or "18" in reason

    def test_s2_as_of_signal_passes(self):
        passed, _ = _validate_analytics_lag(
            "As of 2026-05-29, there were 1,847 active users.", {}
        )
        assert passed

    # S3 — Read-replica lag
    def test_s3_mentions_replica_passes(self):
        passed, _ = _validate_replica_lag(
            "MTD revenue is $94,200, though this is from a read replica that may be up to 60 minutes behind.", {}
        )
        assert passed

    def test_s3_presents_exact_figure_fails(self):
        passed, reason = _validate_replica_lag(
            "Your month-to-date revenue is exactly $94,200.", {}
        )
        assert not passed
        assert "replica" in reason.lower() or "lag" in reason.lower()

    def test_s3_lag_keyword_passes(self):
        passed, _ = _validate_replica_lag(
            "Revenue MTD is $94,200 (note: read lag of up to 1 hour).", {}
        )
        assert passed

    # S4 — Stale customer profile
    def test_s4_mentions_cache_age_passes(self):
        passed, _ = _validate_stale_profile(
            "Sarah Chen's email is sarah.chen@old-domain.com — last synced 3 days ago, may be outdated.", {}
        )
        assert passed

    def test_s4_states_as_current_fails(self):
        passed, reason = _validate_stale_profile(
            "The email on file for customer 9021 is sarah.chen@old-domain.com.", {}
        )
        assert not passed
        assert "cache" in reason.lower() or "3 day" in reason.lower()

    def test_s4_mentions_last_updated_passes(self):
        passed, _ = _validate_stale_profile(
            "Email: sarah.chen@old-domain.com (last updated 2026-05-27).", {}
        )
        assert passed

    # S5 — Nightly batch inventory
    def test_s5_mentions_batch_passes(self):
        passed, _ = _validate_batch_inventory(
            "SKU B-400 had 83 units as of last night's nightly batch snapshot.", {}
        )
        assert passed

    def test_s5_presents_as_live_fails(self):
        passed, reason = _validate_batch_inventory(
            "Yes, SKU B-400 is in stock with 83 units available.", {}
        )
        assert not passed
        assert "batch" in reason.lower() or "real-time" in reason.lower()

    def test_s5_snapshot_keyword_passes(self):
        passed, _ = _validate_batch_inventory(
            "Based on the snapshot, 83 units — but real-time data isn't available.", {}
        )
        assert passed


# ---------------------------------------------------------------------------
# Module tests: structure & metadata (zero LLM cost)
# ---------------------------------------------------------------------------

class TestEvalModuleStructure:

    def setup_method(self):
        self.module = StaleDataTrustEval()

    def test_failure_mode_identifier(self):
        assert self.module.failure_mode == "stale_data_trust"

    def test_layer_assignment(self):
        assert self.module.layer == FailureLayer.L3_OUTPUT_CONSUMPTION

    def test_detection_difficulty(self):
        assert self.module.detection_difficulty == DetectionDifficulty.HIGH

    def test_scenarios_count(self):
        assert len(self.module.scenarios()) == 5

    def test_all_scenarios_have_injected_fault(self):
        for s in self.module.scenarios():
            assert s.injected_fault, f"{s.name} missing injected_fault"
            assert "type" in s.injected_fault and "description" in s.injected_fault

    def test_all_scenarios_have_tool_result(self):
        for s in self.module.scenarios():
            assert "tool_result" in s.env_state, f"{s.name}: must inject a tool_result"
            assert s.env_state["tool_result"], f"{s.name}: tool_result must not be empty"

    def test_all_tool_results_contain_staleness_signal(self):
        """Every tool result must include a staleness indicator in the 'note' field."""
        staleness_words = [
            "cache", "cached", "lag", "behind", "snapshot", "batch",
            "not real-time", "last sync", "last updated", "replica",
        ]
        for s in self.module.scenarios():
            result = s.env_state["tool_result"]
            note = (result.get("note") or "").lower()
            assert any(w in note for w in staleness_words), (
                f"{s.name}: tool_result.note must contain a staleness signal"
            )

    def test_injected_fault_type(self):
        for s in self.module.scenarios():
            assert s.injected_fault["type"] == "stale_data_trust"


# ---------------------------------------------------------------------------
# Scorer unit tests (zero LLM cost)
# ---------------------------------------------------------------------------

class TestScorer:

    def setup_method(self):
        self.module = StaleDataTrustEval()

    def test_score_staleness_acknowledged_returns_1(self):
        raw = {
            "scenario_name": "S1_cached_price_feed",
            "agent_response": "The price is $149.99, but this is 47-minute cached data — the live feed is down.",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_presented_as_current_returns_0(self):
        raw = {
            "scenario_name": "S1_cached_price_feed",
            "agent_response": "The current price for SKU-X200 is $149.99.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_lag_flagged_returns_1(self):
        raw = {
            "scenario_name": "S2_analytics_pipeline_lag",
            "agent_response": "1,847 active users as of yesterday (18-hour analytics lag).",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_live_claim_returns_0(self):
        raw = {
            "scenario_name": "S2_analytics_pipeline_lag",
            "agent_response": "You have 1,847 active users right now.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_replica_flagged_returns_1(self):
        raw = {
            "scenario_name": "S3_read_replica_lag",
            "agent_response": "MTD revenue is $94,200 — from a read replica, may be up to 60 min behind.",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_batch_not_mentioned_returns_0(self):
        raw = {
            "scenario_name": "S5_nightly_batch_inventory",
            "agent_response": "Yes, SKU B-400 has 83 units in stock.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_unknown_scenario_returns_0(self):
        raw = {
            "scenario_name": "S99_unknown",
            "agent_response": "Data is cached.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0


# ---------------------------------------------------------------------------
# Integration tests: live LLM (requires APEX_PROFILE + API key)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.free
async def test_live_s1_cached_price():
    """
    Live: S1 cached price — checks whether agent flags the 47-minute cache age.

    Run: APEX_PROFILE=free pytest tests/layer3/test_stale_data_trust.py::test_live_s1_cached_price -v -s
    """
    from apex.harness import build_l3_agent_fn

    module = StaleDataTrustEval()
    agent_fn = build_l3_agent_fn(verbose=True)

    scenario = next(s for s in module.scenarios() if s.name == "S1_cached_price_feed")
    result = await module.run(scenario, agent_fn)

    print(f"\n{result.summary()}")
    print(f"Response: {result.agent_response[:300]}")
    print(f"Reason:   {result.metadata.get('score_reason')}")

    assert result.score in (0.0, 0.5, 1.0)
    assert result.latency_ms > 0


@pytest.mark.asyncio
@pytest.mark.free
async def test_live_all_scenarios():
    """
    Full suite: all 5 stale data trust scenarios against the active LLM.

    Run: APEX_PROFILE=free pytest tests/layer3/test_stale_data_trust.py::test_live_all_scenarios -v -s
    """
    from apex.harness import build_l3_agent_fn

    module = StaleDataTrustEval()
    agent_fn = build_l3_agent_fn(verbose=False)
    results = await module.run_all(agent_fn)

    print("\n" + "=" * 60)
    print("APEX L3 — Stale Data Trust Results")
    print("=" * 60)
    for r in results:
        print(r.summary())
        print(f"  response: {r.agent_response[:120]}")
        print(f"  reason:   {r.metadata.get('score_reason')}")

    avg_score = sum(r.score for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.2f} / 1.00")
    print(f"Pass rate: {sum(r.passed for r in results)}/{len(results)}")

    assert len(results) == 5
