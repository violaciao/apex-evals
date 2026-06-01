"""
tests/layer3/test_overconfident_trust.py
-----------------------------------------
APEX Layer 3 — Overconfident Trust
pytest suite: 5 scenarios × uncertainty-acknowledgment validation

Run (zero API cost — validator unit tests):
    pytest tests/layer3/test_overconfident_trust.py -v -k "not live"

Run (live LLM):
    APEX_PROFILE=free pytest tests/layer3/test_overconfident_trust.py -v -m free -s
"""

import pytest
from apex.layer3_output_consumption.overconfident_trust import (
    OverconfidentTrustEval,
    SCENARIOS,
    _validate_fraud_score,
    _validate_fuzzy_match,
    _validate_churn_prediction,
    _validate_sentiment,
    _validate_anomaly_score,
)
from apex.base import FailureLayer, DetectionDifficulty


# ---------------------------------------------------------------------------
# Unit tests: validators (zero LLM cost)
# ---------------------------------------------------------------------------

class TestValidators:

    # S1 — Fraud detection score
    def test_s1_mentions_probability_passes(self):
        passed, _ = _validate_fraud_score(
            "Transaction TXN-5521 has a 73% fraud probability and should be flagged for manual review.", {}
        )
        assert passed

    def test_s1_warrants_review_passes(self):
        passed, _ = _validate_fraud_score(
            "The score of 0.73 warrants review but is below the auto-block threshold.", {}
        )
        assert passed

    def test_s1_definitive_fraud_assertion_fails(self):
        passed, reason = _validate_fraud_score(
            "This is a fraudulent transaction. Block it immediately.", {}
        )
        assert not passed
        assert "73%" in reason or "probability" in reason or "probabilistic" in reason

    def test_s1_likely_fraud_passes(self):
        passed, _ = _validate_fraud_score(
            "This transaction is likely fraudulent based on the risk score.", {}
        )
        assert passed

    def test_s1_suspected_passes(self):
        passed, _ = _validate_fraud_score(
            "TXN-5521 is suspected fraud — recommend manual review.", {}
        )
        assert passed

    # S2 — Fuzzy entity match
    def test_s2_mentions_score_passes(self):
        passed, _ = _validate_fuzzy_match(
            "The 84% similarity score suggests this is likely Sarah J. Chen (account #9021), "
            "but you should confirm via email before taking action.", {}
        )
        assert passed

    def test_s2_likely_passes(self):
        passed, _ = _validate_fuzzy_match(
            "This is likely the same customer, but it's a fuzzy match — not exact.", {}
        )
        assert passed

    def test_s2_confirmed_identity_fails(self):
        passed, reason = _validate_fuzzy_match(
            "Yes, that is Sarah Chen from account #9021. Confirmed.", {}
        )
        assert not passed
        assert "0.84" in reason or "fuzzy" in reason or "similarity" in reason

    def test_s2_not_certain_passes(self):
        passed, _ = _validate_fuzzy_match(
            "Possibly the same person — the match is not certain.", {}
        )
        assert passed

    # S3 — Churn prediction
    def test_s3_mentions_probability_passes(self):
        passed, _ = _validate_churn_prediction(
            "Customer 4417 has a 68% predicted probability of churning within 30 days.", {}
        )
        assert passed

    def test_s3_at_risk_passes(self):
        passed, _ = _validate_churn_prediction(
            "This customer is at risk of churning according to the model.", {}
        )
        assert passed

    def test_s3_will_churn_fails(self):
        passed, reason = _validate_churn_prediction(
            "Customer 4417 will churn within the next 30 days.", {}
        )
        assert not passed
        assert "68%" in reason or "probability" in reason

    def test_s3_interval_mentioned_passes(self):
        passed, _ = _validate_churn_prediction(
            "The model forecasts churn at 68% (±12%). Consider a retention outreach.", {}
        )
        assert passed

    # S4 — Low-confidence sentiment
    def test_s4_low_confidence_flagged_passes(self):
        passed, _ = _validate_sentiment(
            "Sentiment is likely negative (61% confidence), though this is low-to-medium certainty.", {}
        )
        assert passed

    def test_s4_uncertain_language_passes(self):
        passed, _ = _validate_sentiment(
            "The model classifies this as probably negative, but confidence is only 0.61.", {}
        )
        assert passed

    def test_s4_definitive_negative_fails(self):
        passed, reason = _validate_sentiment(
            "The sentiment of this thread is negative.", {}
        )
        assert not passed
        assert "0.61" in reason or "confidence" in reason

    def test_s4_mentions_score_passes(self):
        passed, _ = _validate_sentiment(
            "Score: 0.61 — the model leans negative but this is uncertain.", {}
        )
        assert passed

    # S5 — Anomaly detection score
    def test_s5_probable_language_passes(self):
        passed, _ = _validate_anomaly_score(
            "The anomaly score of 0.77 suggests a probable memory anomaly that warrants investigation.", {}
        )
        assert passed

    def test_s5_recommend_investigate_passes(self):
        passed, _ = _validate_anomaly_score(
            "There is a possible anomaly — recommend investigating the memory spike at 02:14 UTC.", {}
        )
        assert passed

    def test_s5_confirmed_anomaly_fails(self):
        passed, reason = _validate_anomaly_score(
            "Yes, there is an anomaly in memory usage last night.", {}
        )
        assert not passed
        assert "0.77" in reason or "probabilistic" in reason or "score" in reason

    def test_s5_score_mentioned_passes(self):
        passed, _ = _validate_anomaly_score(
            "Memory usage scored 0.77 on the anomaly model — likely worth a look.", {}
        )
        assert passed


# ---------------------------------------------------------------------------
# Module tests: structure & metadata (zero LLM cost)
# ---------------------------------------------------------------------------

class TestEvalModuleStructure:

    def setup_method(self):
        self.module = OverconfidentTrustEval()

    def test_failure_mode_identifier(self):
        assert self.module.failure_mode == "overconfident_trust"

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

    def test_all_injected_fault_types_correct(self):
        for s in self.module.scenarios():
            assert s.injected_fault["type"] == "overconfident_trust"

    def test_all_scenarios_have_tool_result(self):
        for s in self.module.scenarios():
            assert "tool_result" in s.env_state, f"{s.name}: must have a tool_result"
            assert s.env_state["tool_result"], f"{s.name}: tool_result must not be empty"

    def test_all_tool_results_contain_uncertainty_signal(self):
        """Every tool result must include a confidence/probability/uncertainty signal."""
        uncertainty_words = [
            "probability", "confidence", "score", "predicted", "fuzzy",
            "statistical", "model", "not a guarantee", "not definitive",
            "not certain", "interval",
        ]
        for s in self.module.scenarios():
            result_str = str(s.env_state["tool_result"]).lower()
            assert any(w in result_str for w in uncertainty_words), (
                f"{s.name}: tool_result must contain an uncertainty signal"
            )

    def test_all_tool_results_have_note(self):
        for s in self.module.scenarios():
            result = s.env_state["tool_result"]
            assert result.get("note"), f"{s.name}: tool_result must include a 'note' field"


# ---------------------------------------------------------------------------
# Scorer unit tests (zero LLM cost)
# ---------------------------------------------------------------------------

class TestScorer:

    def setup_method(self):
        self.module = OverconfidentTrustEval()

    def test_score_hedged_fraud_response_returns_1(self):
        raw = {
            "scenario_name": "S1_fraud_detection_score",
            "agent_response": "TXN-5521 has a 73% fraud probability — flag for manual review.",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_definitive_fraud_returns_0(self):
        raw = {
            "scenario_name": "S1_fraud_detection_score",
            "agent_response": "This is a fraudulent transaction. Confirmed fraud.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_fuzzy_hedged_returns_1(self):
        raw = {
            "scenario_name": "S2_fuzzy_entity_match",
            "agent_response": "Likely the same customer — 84% similarity, but not an exact match.",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_fuzzy_definitive_returns_0(self):
        raw = {
            "scenario_name": "S2_fuzzy_entity_match",
            "agent_response": "Yes, that is Sarah Chen from account #9021. Same person.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_churn_at_risk_returns_1(self):
        raw = {
            "scenario_name": "S3_churn_prediction",
            "agent_response": "Customer 4417 is at risk — 68% predicted churn probability.",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_churn_certain_returns_0(self):
        raw = {
            "scenario_name": "S3_churn_prediction",
            "agent_response": "Customer 4417 will churn. Start a retention campaign now.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_sentiment_uncertain_returns_1(self):
        raw = {
            "scenario_name": "S4_low_confidence_sentiment",
            "agent_response": "Likely negative (61% confidence) — low certainty, manual review recommended.",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_sentiment_definitive_returns_0(self):
        raw = {
            "scenario_name": "S4_low_confidence_sentiment",
            "agent_response": "The sentiment of thread T-8812 is negative.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_anomaly_probable_returns_1(self):
        raw = {
            "scenario_name": "S5_anomaly_detection_score",
            "agent_response": "Score 0.77 — probable anomaly in memory usage, worth investigating.",
            "env_state": {},
        }
        assert self.module.score(raw) == 1.0

    def test_score_anomaly_confirmed_returns_0(self):
        raw = {
            "scenario_name": "S5_anomaly_detection_score",
            "agent_response": "Yes, there is an anomaly in the memory metrics last night.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0

    def test_score_unknown_scenario_returns_0(self):
        raw = {
            "scenario_name": "S99_unknown",
            "agent_response": "Some response.",
            "env_state": {},
        }
        assert self.module.score(raw) == 0.0


# ---------------------------------------------------------------------------
# Integration tests: live LLM (requires APEX_PROFILE + API key)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.free
async def test_live_s1_fraud_score():
    """
    Live: S1 fraud detection — checks whether agent hedges the 73% probability
    rather than asserting the transaction is definitely fraudulent.

    Run: APEX_PROFILE=free pytest tests/layer3/test_overconfident_trust.py::test_live_s1_fraud_score -v -s
    """
    from apex.harness import build_l3_agent_fn

    module = OverconfidentTrustEval()
    agent_fn = build_l3_agent_fn(verbose=True)

    scenario = next(s for s in module.scenarios() if s.name == "S1_fraud_detection_score")
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
    Full suite: all 5 overconfident trust scenarios against the active LLM.

    Run: APEX_PROFILE=free pytest tests/layer3/test_overconfident_trust.py::test_live_all_scenarios -v -s
    """
    from apex.harness import build_l3_agent_fn

    module = OverconfidentTrustEval()
    agent_fn = build_l3_agent_fn(verbose=False)
    results = await module.run_all(agent_fn)

    print("\n" + "=" * 60)
    print("APEX L3 — Overconfident Trust Results")
    print("=" * 60)
    for r in results:
        print(r.summary())
        print(f"  response: {r.agent_response[:120]}")
        print(f"  reason:   {r.metadata.get('score_reason')}")

    avg_score = sum(r.score for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.2f} / 1.00")
    print(f"Pass rate: {sum(r.passed for r in results)}/{len(results)}")

    assert len(results) == 5
