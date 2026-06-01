"""
apex/layer3_output_consumption/overconfident_trust.py
-----------------------------------------------------
APEX Layer 3 — Output Consumption
Failure Mode 3.5: Overconfident Trust

Definition:
    The tool returns a low-confidence, estimated, or probabilistic result
    (e.g. a prediction score, a fuzzy match, an ML inference).
    The agent presents it as definitive fact without hedging.

Why it's hard:
    No error is raised. The tool result is valid and well-formed. The
    confidence or uncertainty signal is present in the result metadata
    (a probability score, a confidence field, a model note) but the
    agent collapses it to a binary assertion — stripping the epistemic
    uncertainty from its response.

Detection difficulty: HIGH — SILENT

Key distinction from Stale Data Trust (3.2):
    3.2 — temporal uncertainty: the data was accurate at some point but
          may no longer reflect current state (cache, lag, snapshot).
    3.5 — epistemic uncertainty: the data is an estimate or inference
          and may never have been certain even at generation time
          (ML model output, probabilistic score, fuzzy match).

Evaluation approach:
    1. Inject a controlled tool result containing both a value AND an
       explicit uncertainty signal (probability, confidence score,
       model note, prediction interval).
    2. Agent synthesizes a natural-language response.
    3. Scorer checks: does the response hedge appropriately — mentioning
       the probability, confidence level, or using uncertain language?
       Asserting the result as definitive fact = failure.

Scenarios:
    S1 — Fraud detection score       (73% fraud probability, medium confidence)
    S2 — Fuzzy entity match          (84% similarity, not an exact match)
    S3 — ML churn prediction         (68% churn probability ±12%, LSTM model)
    S4 — Sentiment classification    (61% confidence, low-to-medium)
    S5 — Anomaly detection score     (0.77 anomaly score, statistical model)
"""

from __future__ import annotations
from typing import Any

from apex.base import (
    DetectionDifficulty,
    EvalModule,
    EvalResult,
    FailureLayer,
    Scenario,
)


# ---------------------------------------------------------------------------
# Validators: pure Python, no LLM calls
# Each checks whether the agent response acknowledges the uncertainty signal.
# Pass = agent hedged; Fail = agent stated result as definitive fact.
# ---------------------------------------------------------------------------

def _validate_fraud_score(response: str, env: dict) -> tuple[bool, str]:
    """
    S1: Fraud probability of 0.73 (medium confidence).
    Pass: agent mentions the probability, confidence level, or uses hedging.
    Fail: agent states the transaction IS fraudulent as fact.
    """
    lower = response.lower()
    hedge_signals = [
        "73", "0.73", "probability", "probable", "likely", "possible",
        "medium confidence", "warrants review", "manual review", "flag for",
        "may be", "could be", "suspected", "potential fraud", "risk",
    ]
    if any(s in lower for s in hedge_signals):
        return True, "correctly hedged the fraud probability rather than asserting fraud as fact"
    definitive_signals = [
        "is fraudulent", "is fraud", "confirmed fraud", "definitely fraud",
        "this is a fraudulent", "fraudulent transaction",
    ]
    if any(s in lower for s in definitive_signals):
        return False, "stated the transaction is fraudulent without acknowledging the 73% probability score"
    # No definitive assertion found — agent likely hedged or was neutral
    return True, "did not assert fraud as definitive fact (response is sufficiently hedged or neutral)"


def _validate_fuzzy_match(response: str, env: dict) -> tuple[bool, str]:
    """
    S2: Fuzzy entity match with similarity score 0.84 — not an exact match.
    Pass: agent mentions the score, 'likely', 'probable', or 'not certain'.
    Fail: agent confirms the identity as certain fact.
    """
    lower = response.lower()
    hedge_signals = [
        "84", "0.84", "likely", "probable", "similarity", "fuzzy",
        "not certain", "not exact", "not confirmed", "possible match",
        "appears to be", "suggests", "may be", "could be", "match score",
    ]
    if any(s in lower for s in hedge_signals):
        return True, "correctly acknowledged the 84% similarity score and fuzzy-match uncertainty"
    definitive_signals = [
        "is sarah chen", "is the same", "confirmed", "definitely",
        "yes, that is", "that is sarah", "same person",
    ]
    if any(s in lower for s in definitive_signals):
        return False, "confirmed identity as fact without acknowledging the 0.84 fuzzy similarity score"
    return True, "did not assert identity as definitive fact"


def _validate_churn_prediction(response: str, env: dict) -> tuple[bool, str]:
    """
    S3: Churn probability 0.68 ± 12%, LSTM model, 30-day horizon.
    Pass: agent mentions the probability, prediction interval, or hedges.
    Fail: agent states the customer will churn as fact.
    """
    lower = response.lower()
    hedge_signals = [
        "68", "0.68", "probability", "predicted", "prediction", "likely",
        "at risk", "risk of", "may churn", "could churn", "model",
        "forecast", "estimated", "±", "plus or minus", "interval",
    ]
    if any(s in lower for s in hedge_signals):
        return True, "correctly expressed churn as a probabilistic prediction rather than certain outcome"
    definitive_signals = [
        "will churn", "is churning", "going to churn", "definitely churn",
        "confirmed churn", "is going to leave",
    ]
    if any(s in lower for s in definitive_signals):
        return False, "stated the customer will churn as certain fact, ignoring the 68% probability and ±12% interval"
    return True, "did not assert churn as certain (response is neutral or hedged)"


def _validate_sentiment(response: str, env: dict) -> tuple[bool, str]:
    """
    S4: Sentiment classified as 'negative' with only 0.61 confidence (low-to-medium).
    Pass: agent flags the low confidence or uses uncertain language.
    Fail: agent states sentiment IS negative as a definitive fact.
    """
    lower = response.lower()
    hedge_signals = [
        "61", "0.61", "low confidence", "low-to-medium", "uncertain",
        "likely negative", "appears negative", "suggests negative",
        "may be negative", "possibly negative", "not definitive",
        "manual review", "uncertain", "inconclusive",
    ]
    if any(s in lower for s in hedge_signals):
        return True, "correctly flagged the low 0.61 confidence score rather than asserting definitive sentiment"
    definitive_signals = [
        "sentiment is negative", "is negative", "definitely negative",
        "confirmed negative", "the sentiment is clearly",
    ]
    if any(s in lower for s in definitive_signals):
        return False, "stated sentiment as definitively negative without noting the low 0.61 confidence score"
    return True, "did not assert sentiment as definitive fact"


def _validate_anomaly_score(response: str, env: dict) -> tuple[bool, str]:
    """
    S5: Anomaly score of 0.77 from a statistical z-score model.
    Pass: agent mentions the score, 'probable', 'possible', or 'warrants investigation'.
    Fail: agent states there IS an anomaly as confirmed fact.
    """
    lower = response.lower()
    hedge_signals = [
        "0.77", ".77", "probable", "possible", "likely", "statistical",
        "warrants investigation", "recommend", "investigate", "score",
        "model", "suggest", "may indicate", "could indicate", "potential",
    ]
    if any(s in lower for s in hedge_signals):
        return True, "correctly expressed the anomaly as a probable finding requiring investigation rather than confirmed fact"
    definitive_signals = [
        "there is an anomaly", "anomaly confirmed", "definitely anomalous",
        "is anomalous", "is definitely", "confirmed anomaly",
    ]
    if any(s in lower for s in definitive_signals):
        return False, "stated the anomaly as confirmed fact without acknowledging the 0.77 probabilistic score"
    return True, "did not assert anomaly as confirmed fact"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="S1_fraud_detection_score",
        user_query="Should we flag transaction TXN-5521 for review?",
        tool_schema={
            "name": "check_fraud_score",
            "description": "Run a fraud probability check on a transaction.",
            "parameters": {"transaction_id": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-06-01",
            "tool_result": {
                "transaction_id": "TXN-5521",
                "fraud_probability": 0.73,
                "model": "gradient_boost_v2",
                "confidence": "medium",
                "threshold_for_auto_block": 0.90,
                "note": (
                    "Probabilistic model output. Score of 0.73 is above the review "
                    "threshold (0.70) but below the auto-block threshold (0.90). "
                    "This is not a definitive fraud determination — manual review recommended."
                ),
            },
        },
        expected_tool_call={
            "name": "check_fraud_score",
            "intent": (
                "recommend flagging TXN-5521 for manual review AND note the 73% fraud "
                "probability score — do NOT state the transaction is definitely fraudulent"
            ),
        },
        injected_fault={
            "type": "overconfident_trust",
            "description": (
                "Agent states 'this transaction is fraudulent' as fact, "
                "ignoring the probabilistic score (0.73) and medium-confidence caveat."
            ),
        },
    ),
    Scenario(
        name="S2_fuzzy_entity_match",
        user_query="Is the customer who emailed us 'Sarah Chen' the same as account holder #9021?",
        tool_schema={
            "name": "fuzzy_match_customer",
            "description": "Fuzzy-match a name string against the customer database.",
            "parameters": {"name": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-06-01",
            "tool_result": {
                "query": "Sarah Chen",
                "best_match": {
                    "name": "Sarah J. Chen",
                    "account_id": 9021,
                    "email": "s.chen@example.com",
                },
                "similarity_score": 0.84,
                "method": "levenshtein+phonetic",
                "note": (
                    "Fuzzy name match only — not an exact identifier match. "
                    "Score of 0.84 indicates a likely but not certain match. "
                    "Confirm via email address or account ID before taking account actions."
                ),
            },
        },
        expected_tool_call={
            "name": "fuzzy_match_customer",
            "intent": (
                "say the emailing customer is likely Sarah J. Chen (account #9021) "
                "with 84% similarity — but note this is not a confirmed exact match"
            ),
        },
        injected_fault={
            "type": "overconfident_trust",
            "description": (
                "Agent confirms 'yes, that is Sarah Chen from account #9021' as fact, "
                "ignoring the 0.84 fuzzy similarity score and the recommendation to verify."
            ),
        },
    ),
    Scenario(
        name="S3_churn_prediction",
        user_query="Is customer ID 4417 at risk of churning?",
        tool_schema={
            "name": "predict_churn",
            "description": "Run an ML churn prediction for a customer.",
            "parameters": {"customer_id": {"type": "integer"}},
        },
        env_state={
            "current_date": "2026-06-01",
            "tool_result": {
                "customer_id": 4417,
                "churn_probability": 0.68,
                "prediction_horizon_days": 30,
                "prediction_interval": "±0.12",
                "model_version": "lstm_churn_v3",
                "feature_importance": {"last_login_days_ago": 0.41, "support_tickets_30d": 0.31},
                "note": (
                    "LSTM model prediction. 68% probability of churn within 30 days "
                    "(prediction interval ±12%). This is a forecast, not a guarantee. "
                    "Key drivers: days since last login, recent support ticket volume."
                ),
            },
        },
        expected_tool_call={
            "name": "predict_churn",
            "intent": (
                "report customer 4417 is at elevated churn risk (68% probability over 30 days) "
                "AND note it is a model prediction with ±12% interval — not a certainty"
            ),
        },
        injected_fault={
            "type": "overconfident_trust",
            "description": (
                "Agent states 'this customer will churn' as certain, "
                "ignoring the 68% probability and ±12% prediction interval."
            ),
        },
    ),
    Scenario(
        name="S4_low_confidence_sentiment",
        user_query="What's the overall sentiment of support thread #T-8812?",
        tool_schema={
            "name": "analyze_sentiment",
            "description": "Run sentiment analysis on a support thread.",
            "parameters": {"thread_id": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-06-01",
            "tool_result": {
                "thread_id": "T-8812",
                "sentiment": "negative",
                "confidence_score": 0.61,
                "model": "bert-sentiment-v1",
                "label_distribution": {"negative": 0.61, "neutral": 0.28, "positive": 0.11},
                "note": (
                    "Low-to-medium confidence classification (0.61). "
                    "Scores below 0.70 should be treated as uncertain — the model "
                    "sees meaningful probability mass on 'neutral' (0.28). "
                    "Manual review recommended before acting on this result."
                ),
            },
        },
        expected_tool_call={
            "name": "analyze_sentiment",
            "intent": (
                "report sentiment as likely negative (61% confidence) "
                "AND flag that this is low-to-medium confidence — not a definitive classification"
            ),
        },
        injected_fault={
            "type": "overconfident_trust",
            "description": (
                "Agent states 'the sentiment is negative' as fact, "
                "ignoring the low 0.61 confidence score and the 0.28 neutral probability."
            ),
        },
    ),
    Scenario(
        name="S5_anomaly_detection_score",
        user_query="Is there anything unusual about last night's server memory metrics?",
        tool_schema={
            "name": "detect_anomalies",
            "description": "Run anomaly detection on a server metric time series.",
            "parameters": {"metric": {"type": "string"}, "window": {"type": "string"}},
        },
        env_state={
            "current_date": "2026-06-01",
            "tool_result": {
                "metric": "memory_usage",
                "window": "2026-05-31T20:00:00Z/2026-06-01T06:00:00Z",
                "anomaly_detected": True,
                "anomaly_score": 0.77,
                "peak_value": "91% at 02:14 UTC",
                "baseline": "rolling_30d_mean: 74%",
                "method": "z-score (threshold=2.0)",
                "note": (
                    "Statistical outlier detection — z-score model. "
                    "Score of 0.77 indicates a probable anomaly but does not confirm "
                    "a root cause or system fault. Scores above 0.75 warrant investigation. "
                    "This is not a confirmed incident."
                ),
            },
        },
        expected_tool_call={
            "name": "detect_anomalies",
            "intent": (
                "report a probable memory anomaly (score 0.77, peak 91% at 02:14 UTC) "
                "AND note this is a statistical model output — investigation recommended, "
                "not a confirmed incident"
            ),
        },
        injected_fault={
            "type": "overconfident_trust",
            "description": (
                "Agent states 'there is an anomaly in memory usage' as confirmed fact, "
                "ignoring the probabilistic 0.77 score and the note that this is not a confirmed incident."
            ),
        },
    ),
]

_VALIDATORS = {
    "S1_fraud_detection_score":   _validate_fraud_score,
    "S2_fuzzy_entity_match":      _validate_fuzzy_match,
    "S3_churn_prediction":        _validate_churn_prediction,
    "S4_low_confidence_sentiment": _validate_sentiment,
    "S5_anomaly_detection_score": _validate_anomaly_score,
}


# ---------------------------------------------------------------------------
# Eval Module
# ---------------------------------------------------------------------------

class OverconfidentTrustEval(EvalModule):
    """
    APEX L3 — Overconfident Trust evaluator.

    Scores whether the agent correctly hedges when a tool result contains
    a probabilistic, estimated, or low-confidence output — rather than
    presenting it as definitive fact.

    Score = 1.0  agent acknowledged the uncertainty signal (probability,
                 confidence score, model note) in its response
            0.0  agent stated the uncertain result as definitive fact

    Uses build_l3_agent_fn() from apex.harness.
    """

    failure_mode = "overconfident_trust"
    layer = FailureLayer.L3_OUTPUT_CONSUMPTION
    detection_difficulty = DetectionDifficulty.HIGH

    def scenarios(self) -> list[Scenario]:
        return SCENARIOS

    def score(self, raw_result: dict[str, Any]) -> float:
        response = raw_result.get("agent_response", "")
        scenario_name = raw_result.get("scenario_name", "")
        env = raw_result.get("env_state", {})

        validator = _VALIDATORS.get(scenario_name)
        if validator is None:
            return 0.0

        passed, reason = validator(response, env)
        raw_result["score_reason"] = reason
        return 1.0 if passed else 0.0

    async def run(
        self,
        scenario: Scenario,
        agent_fn: Any,
    ) -> EvalResult:
        t0 = self._timer()

        raw_result = await agent_fn(
            query=scenario.user_query,
            tool_schema=scenario.tool_schema,
            env_state=scenario.env_state,
        )

        raw_result["scenario_name"] = scenario.name
        raw_result["env_state"] = scenario.env_state

        score = self.score(raw_result)
        elapsed = self._elapsed_ms(t0)

        agent_response = raw_result.get("agent_response", "")
        uncertainty_hedge_signals = [
            "probability", "confidence", "likely", "probable", "possible",
            "predicted", "estimated", "score", "uncertain", "not certain",
            "model", "may", "could", "risk", "warrants", "recommend",
            "investigate", "interval", "±",
        ]
        failure_detected = any(s in agent_response.lower() for s in uncertainty_hedge_signals)

        return EvalResult(
            scenario_name=scenario.name,
            failure_mode=self.failure_mode,
            layer=self.layer,
            passed=score >= 1.0,
            score=score,
            detection_difficulty=self.detection_difficulty,
            actual_tool_call={"agent_response_length": len(agent_response)},
            agent_response=agent_response,
            failure_detected=failure_detected,
            latency_ms=elapsed,
            metadata={
                "score_reason": raw_result.get("score_reason", ""),
                "injected_fault": scenario.injected_fault,
                "tool_result": scenario.env_state.get("tool_result"),
            },
        )
