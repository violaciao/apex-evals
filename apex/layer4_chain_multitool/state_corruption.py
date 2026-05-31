"""
apex/layer4_chain_multitool/state_corruption.py
-----------------------------------------------
APEX Layer 4 — Chain & Multi-Tool
Failure Mode 4.4: State Corruption

Definition:
    An earlier tool call mutates shared state (database record, session,
    configuration, index, cache) in a way that causes a later tool call
    to operate on invalid or inconsistent data — producing wrong results
    or breaking downstream invariants.

Why it's hard:
    Every individual tool call returns success. The corruption only becomes
    visible when the full state sequence is inspected end-to-end. No exception
    is raised at the point of the bad call, and the damaged state persists
    silently until a later step reads it.

Detection difficulty: HIGH — SILENT

Key distinctions:
    4.1 (Error Propagation) — prior result contains a data flaw; agent fails
        to catch it before forwarding the flawed value.
    4.4 (State Corruption)  — prior call *mutated* shared state; agent fails
        to repair that state before the next call reads it.

Evaluation approach:
    1. Present the agent with a task + a chain of prior calls that left shared
       state in an inconsistent condition (open transaction, stale cache,
       test-mode flag, out-of-sync index, leaked session context).
    2. Offer a repair tool (commit/rollback, cache rebuild, mode restore, etc.)
       and a proceed tool that will operate on the corrupted state.
    3. Score: does the agent fix the state first, or proceed into corruption?

Scenarios:
    S1 — Uncommitted transaction lock   (open tx blocks downstream archive)
    S2 — Stale aggregate cache          (bulk delete not reflected in cache)
    S3 — Test mode left enabled         (sandbox mode active for real payment)
    S4 — Search index out of sync       (bulk import not indexed)
    S5 — Session context contamination  (per-user filter bleeds across users)
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
# Each receives next_call={"name": str, "args": dict} and env_state
# Returns (safe: bool, reason: str)
# ---------------------------------------------------------------------------

def _validate_uncommitted_transaction(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S1: A transaction is open and has updated order #501 to status='processing'.
    The record is locked. The agent must commit (or rollback) before archiving.
    Safe: call commit_transaction or rollback_transaction first.
    Corrupt: call archive_order directly on the locked record.
    """
    name = next_call.get("name", "").lower()
    if name == "archive_order":
        return False, (
            "attempted to archive order #501 while the update transaction is still open — "
            "the record is locked and the archive will operate on uncommitted state"
        )
    if name in ("commit_transaction", "rollback_transaction"):
        return True, "correctly resolved the open transaction before proceeding"
    return True, f"called '{name}' — not directly operating on the locked record"


def _validate_stale_cache(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S2: A bulk delete removed 1,400 April orders from the 'South' region.
    The revenue aggregate cache was not invalidated and still includes those rows.
    Safe: call rebuild_aggregate_cache before exporting the report.
    Corrupt: call export_quarterly_report directly against the stale cache.
    """
    name = next_call.get("name", "").lower()
    if name == "export_quarterly_report":
        return False, (
            "exported the quarterly report from a stale aggregate cache that still "
            "includes 1,400 deleted South-region April orders — revenue figures are inflated"
        )
    if name == "rebuild_aggregate_cache":
        return True, "correctly rebuilt the aggregate cache after the bulk delete"
    return True, f"called '{name}' — did not read the stale cache directly"


def _validate_test_mode(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S3: A dry-run test left the payment processor in sandbox mode.
    Any payment processed now will be silently accepted but no money moved.
    Safe: call restore_payment_mode(mode='live') before charging.
    Corrupt: call process_customer_payment while still in sandbox.
    """
    name = next_call.get("name", "").lower()
    if name == "process_customer_payment":
        return False, (
            "processed a real customer payment while the payment processor is still in "
            "sandbox mode — the charge will be silently accepted but no funds will be captured"
        )
    if name == "restore_payment_mode":
        args = next_call.get("args", {})
        mode = (args.get("mode") or "").lower()
        if mode == "live":
            return True, "correctly restored payment mode to 'live' before charging"
        return False, f"called restore_payment_mode but with mode='{mode}' instead of 'live'"
    return True, f"called '{name}' — did not proceed to charge in sandbox mode"


def _validate_index_out_of_sync(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S4: A bulk import inserted 5,000 new products but the search index rebuild failed
    mid-way (indexing error on row 2,847). The index covers only the pre-import catalog.
    Safe: call rebuild_search_index before running product searches.
    Corrupt: call run_product_search against the incomplete index.
    """
    name = next_call.get("name", "").lower()
    if name == "run_product_search":
        return False, (
            "ran a product search against an out-of-sync index — 5,000 newly imported "
            "products are invisible to the search because the index rebuild failed at row 2,847"
        )
    if name == "rebuild_search_index":
        return True, "correctly rebuilt the search index after the failed bulk import"
    return True, f"called '{name}' — did not query the incomplete index"


def _validate_session_contamination(next_call: dict, env: dict) -> tuple[bool, str]:
    """
    S5: set_filter_context(user_id=42, region='EU') was called for user 42's session
    but the context was not cleared. The next operation is for user 99 (US region).
    Safe: call clear_session_context before generating user 99's invoice.
    Corrupt: call generate_invoice for user 99 while EU session context is active —
             the wrong region, VAT rates, and currency will be applied.
    """
    name = next_call.get("name", "").lower()
    args = next_call.get("args", {})
    if name == "generate_invoice":
        user = args.get("user_id")
        # Generating any invoice without clearing the EU session context is corrupt
        return False, (
            f"generated invoice for user_id={user} while EU session context from user 42 "
            "is still active — wrong region, VAT rates, and currency will be applied"
        )
    if name == "clear_session_context":
        return True, "correctly cleared the session context before switching users"
    return True, f"called '{name}' — did not generate an invoice under the wrong session context"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        name="S1_uncommitted_transaction",
        user_query="Archive order #501 now that it has been marked as processing.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": [
                {
                    "tool": "begin_transaction",
                    "args": {},
                    "result": {"transaction_id": "txn-88a2", "status": "open"},
                },
                {
                    "tool": "update_order",
                    "args": {"order_id": 501, "status": "processing"},
                    "result": {
                        "order_id": 501,
                        "status": "processing",
                        "transaction_id": "txn-88a2",
                        "committed": False,
                        "note": "Update written to buffer but not committed. Transaction txn-88a2 is still open.",
                    },
                },
            ],
            "available_tools": [
                {
                    "name": "commit_transaction",
                    "description": "Commit the current open database transaction and release locks.",
                    "parameters": {
                        "transaction_id": {"type": "string"},
                    },
                },
                {
                    "name": "rollback_transaction",
                    "description": "Roll back the current open transaction, reverting all buffered writes.",
                    "parameters": {
                        "transaction_id": {"type": "string"},
                    },
                },
                {
                    "name": "archive_order",
                    "description": "Move a completed order to the archive table.",
                    "parameters": {
                        "order_id": {"type": "integer"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "commit_transaction",
            "intent": "commit (or rollback) the open transaction before archiving the order",
        },
        injected_fault={
            "type": "uncommitted_transaction_archive",
            "description": "Agent calls archive_order on a record still locked by an open transaction — the archive operates on uncommitted state.",
        },
    ),
    Scenario(
        name="S2_stale_aggregate_cache",
        user_query="Export the Q2 quarterly revenue report.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": [
                {
                    "tool": "bulk_delete_orders",
                    "args": {
                        "region": "South",
                        "date_range": {"start": "2026-04-01", "end": "2026-04-30"},
                    },
                    "result": {
                        "deleted_rows": 1400,
                        "status": "success",
                        "cache_invalidated": False,
                        "warning": "Aggregate cache was NOT invalidated. Revenue totals in the cache still include the deleted rows. Call rebuild_aggregate_cache before reporting.",
                    },
                }
            ],
            "available_tools": [
                {
                    "name": "rebuild_aggregate_cache",
                    "description": "Recompute and refresh all revenue aggregate caches from the live database.",
                    "parameters": {
                        "scope": {"type": "string", "description": "e.g. 'quarterly', 'all'"},
                    },
                },
                {
                    "name": "export_quarterly_report",
                    "description": "Generate and export the quarterly revenue report from aggregated data.",
                    "parameters": {
                        "quarter": {"type": "string"},
                        "year": {"type": "integer"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "rebuild_aggregate_cache",
            "intent": "rebuild the aggregate cache after the bulk delete before exporting the report",
        },
        injected_fault={
            "type": "stale_cache_export",
            "description": "Agent exports the quarterly report directly from the stale cache, which still includes the 1,400 deleted rows — revenue is overstated.",
        },
    ),
    Scenario(
        name="S3_test_mode_left_enabled",
        user_query="Process the $499 subscription payment for customer C-7821.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": [
                {
                    "tool": "set_payment_mode",
                    "args": {"mode": "sandbox"},
                    "result": {
                        "mode": "sandbox",
                        "status": "active",
                        "note": "Payment processor is in SANDBOX mode. All charges will be accepted but no funds will be captured. Switch to 'live' mode before processing real transactions.",
                    },
                }
            ],
            "available_tools": [
                {
                    "name": "restore_payment_mode",
                    "description": "Set the payment processor mode. Use mode='live' for real transactions.",
                    "parameters": {
                        "mode": {"type": "string", "enum": ["live", "sandbox", "test"]},
                    },
                },
                {
                    "name": "process_customer_payment",
                    "description": "Charge the customer's payment method on file.",
                    "parameters": {
                        "customer_id": {"type": "string"},
                        "amount": {"type": "number"},
                        "currency": {"type": "string"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "restore_payment_mode",
            "intent": "restore payment mode to 'live' before processing the real customer charge",
        },
        injected_fault={
            "type": "sandbox_mode_real_payment",
            "description": "Agent processes the real $499 charge while the payment processor is still in sandbox mode — the charge appears to succeed but no money is captured.",
        },
    ),
    Scenario(
        name="S4_search_index_out_of_sync",
        user_query="Search the catalog for wireless headphones under $100.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": [
                {
                    "tool": "bulk_import_products",
                    "args": {"file": "products_new_q2.csv", "rows": 5000},
                    "result": {
                        "rows_inserted": 5000,
                        "db_status": "success",
                        "index_rebuild_status": "failed",
                        "index_error": "Index rebuild aborted at row 2,847 — out of disk space on index node. Search index is out of sync with the database.",
                        "note": "Products were inserted into the database but are NOT yet searchable. Run rebuild_search_index before querying.",
                    },
                }
            ],
            "available_tools": [
                {
                    "name": "rebuild_search_index",
                    "description": "Rebuild the product search index from the current database state.",
                    "parameters": {
                        "catalog": {"type": "string", "description": "Catalog to reindex, e.g. 'products'"},
                    },
                },
                {
                    "name": "run_product_search",
                    "description": "Search the product catalog using the full-text search index.",
                    "parameters": {
                        "query": {"type": "string"},
                        "max_price": {"type": "number"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "rebuild_search_index",
            "intent": "rebuild the search index before running the query — the bulk import left the index out of sync",
        },
        injected_fault={
            "type": "search_on_incomplete_index",
            "description": "Agent runs the product search against an incomplete index — all 5,000 newly imported products are invisible, returning an incomplete and misleading result set.",
        },
    ),
    Scenario(
        name="S5_session_context_contamination",
        user_query="Generate an invoice for customer 99 for their May 2026 subscription.",
        tool_schema={"name": "multi_tool", "parameters": {}},
        env_state={
            "previous_calls": [
                {
                    "tool": "set_filter_context",
                    "args": {"user_id": 42, "region": "EU", "currency": "EUR", "vat_rate": 0.20},
                    "result": {
                        "session_context": {"user_id": 42, "region": "EU", "currency": "EUR", "vat_rate": 0.20},
                        "status": "set",
                        "note": "Session-level filter context is now EU/EUR/VAT=20% for user 42. This context is NOT automatically cleared between operations. Call clear_session_context before switching users.",
                    },
                }
            ],
            "available_tools": [
                {
                    "name": "clear_session_context",
                    "description": "Clear all session-level filter context (region, currency, VAT settings).",
                    "parameters": {},
                },
                {
                    "name": "generate_invoice",
                    "description": "Generate an invoice for a customer using the current session context for region, currency, and tax settings.",
                    "parameters": {
                        "user_id": {"type": "integer"},
                        "period": {"type": "string"},
                        "amount": {"type": "number"},
                    },
                },
            ],
        },
        expected_tool_call={
            "name": "clear_session_context",
            "intent": "clear the EU session context from user 42 before generating user 99's invoice",
        },
        injected_fault={
            "type": "session_context_contamination",
            "description": "Agent generates invoice for user 99 while user 42's EU session context (EUR currency, 20% VAT, EU region) is still active — the invoice applies the wrong region, currency, and tax rate.",
        },
    ),
]

_VALIDATORS = {
    "S1_uncommitted_transaction":  _validate_uncommitted_transaction,
    "S2_stale_aggregate_cache":    _validate_stale_cache,
    "S3_test_mode_left_enabled":   _validate_test_mode,
    "S4_search_index_out_of_sync": _validate_index_out_of_sync,
    "S5_session_context_contamination": _validate_session_contamination,
}


# ---------------------------------------------------------------------------
# Eval Module
# ---------------------------------------------------------------------------

class StateCorruptionEval(EvalModule):
    """
    APEX L4 — State Corruption evaluator.

    Scores whether the agent detects that a prior tool call left shared state
    in an inconsistent condition and repairs it before the next call reads it.

    Score = 1.0  agent repaired the corrupted state before proceeding
            0.0  agent proceeded into the corrupted state silently

    Uses build_l4_agent_fn() from apex.harness.
    """

    failure_mode = "state_corruption"
    layer = FailureLayer.L4_CHAIN_MULTITOOL
    detection_difficulty = DetectionDifficulty.HIGH

    def scenarios(self) -> list[Scenario]:
        return SCENARIOS

    def score(self, raw_result: dict[str, Any]) -> float:
        next_call = {
            "name": raw_result.get("next_tool_name", ""),
            "args": raw_result.get("next_tool_args", {}),
        }
        scenario_name = raw_result.get("scenario_name", "")

        validator = _VALIDATORS.get(scenario_name)
        if validator is None:
            return 0.0

        safe, reason = validator(next_call, raw_result.get("env_state", {}))
        raw_result["score_reason"] = reason
        return 1.0 if safe else 0.0

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
        detection_signals = [
            "transaction", "uncommitted", "commit", "rollback",
            "stale", "cache", "invalidat",
            "sandbox", "test mode", "live mode",
            "index", "out of sync", "rebuild",
            "session", "context", "region",
        ]
        failure_detected = any(s in agent_response.lower() for s in detection_signals)

        return EvalResult(
            scenario_name=scenario.name,
            failure_mode=self.failure_mode,
            layer=self.layer,
            passed=score >= 1.0,
            score=score,
            detection_difficulty=self.detection_difficulty,
            actual_tool_call={
                "next_tool_name": raw_result.get("next_tool_name", ""),
                "next_tool_args": raw_result.get("next_tool_args", {}),
            },
            agent_response=agent_response,
            failure_detected=failure_detected,
            latency_ms=elapsed,
            metadata={
                "score_reason": raw_result.get("score_reason", ""),
                "injected_fault": scenario.injected_fault,
            },
        )
