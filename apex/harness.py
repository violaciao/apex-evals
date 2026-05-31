"""
apex/harness.py
---------------
LlamaIndex agent harness.
Provides agent_fn compatible with all EvalModule.run() calls.

Usage:
    from apex.harness import build_agent_fn
    agent_fn = build_agent_fn()
    result = await agent_fn(query=..., tool_schema=..., env_state=...)
"""

from __future__ import annotations
import json
import re
from typing import Any

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool

from apex.config import get_llm


_SYSTEM_PROMPT = """You are a data analyst agent. When the user asks a question,
construct the correct SQL query using the query_db tool, then report the result.
Always use the tool — never answer from memory.
The database schema will be provided in the user message."""

_L3_SYSTEM_PROMPT = """You are a data analyst. A tool has already been called and \
returned results. Report the findings to the user accurately and completely, based \
strictly on the data in the tool result. Do not add estimates, projections, or any \
information not explicitly present in the result."""


def _extract_sql(text: str) -> str:
    """Pull SQL out of agent response (handles markdown fences or raw SQL)."""
    # Try ```sql ... ``` block first
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    # Fall back: first SELECT statement found
    select = re.search(r"(SELECT\s+.+?)(?:;|$)", text, re.DOTALL | re.IGNORECASE)
    if select:
        return select.group(1).strip()
    return ""


def build_agent_fn(verbose: bool = False):
    """
    Returns an async callable:
        async (query, tool_schema, env_state) -> {"sql": str, "agent_response": str}

    Uses LlamaIndex LLM (configured via APEX_PROFILE) in a single-turn
    tool-use pattern — sufficient for L2 evals which test argument construction,
    not multi-step chains.
    """
    llm = get_llm()

    async def agent_fn(
        query: str,
        tool_schema: dict[str, Any],
        env_state: dict[str, Any],
    ) -> dict[str, Any]:

        # Build context message with schema + env hints
        env_context = "\n".join(
            f"- {k}: {v}" for k, v in env_state.items() if k != "db_schema"
        )
        db_schema = env_state.get("db_schema", "")

        user_content = f"""
Database schema:
{db_schema}

Environment context:
{env_context}

User question: {query}

Use the query_db tool with a SQL SELECT statement that correctly answers the question.
""".strip()

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_content),
        ]

        # Single tool available: query_db
        # We don't actually execute the SQL — we capture what the agent constructs.
        # This is intentional: L2 evals score argument construction, not execution.
        tools_spec = [
            {
                "name": tool_schema.get("name", "query_db"),
                "description": tool_schema.get("description", "Run SQL against the orders DB"),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL SELECT statement"}
                    },
                    "required": ["sql"],
                },
            }
        ]

        # LlamaIndex chat with tools
        response = await llm.achat(
            messages=messages,
            tools=tools_spec,
        )

        response_text = response.message.content or ""
        additional_kwargs = response.message.additional_kwargs or {}

        # Extract SQL from tool call (preferred) or response text
        sql = ""
        tool_calls = additional_kwargs.get("tool_calls", [])
        if tool_calls:
            try:
                args = tool_calls[0].get("function", {}).get("arguments", "{}")
                parsed = json.loads(args) if isinstance(args, str) else args
                sql = parsed.get("sql", "")
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        # Fallback: parse SQL from text response
        if not sql:
            sql = _extract_sql(response_text)

        if verbose:
            print(f"\n[agent] query: {query}")
            print(f"[agent] sql:   {sql}")
            print(f"[agent] resp:  {response_text[:200]}")

        return {
            "sql": sql,
            "agent_response": response_text,
            "raw_tool_calls": tool_calls,
        }

    return agent_fn


def build_l3_agent_fn(verbose: bool = False):
    """
    Returns an async callable for L3 (Output Consumption) evals:
        async (query, tool_schema, env_state) -> {"agent_response": str, "tool_result": dict}

    Injects env_state["tool_result"] as the pre-computed tool output.
    The agent synthesizes a natural-language response from it.
    No tool call is made — we evaluate synthesis faithfulness, not construction.
    """
    llm = get_llm()

    async def agent_fn(
        query: str,
        tool_schema: dict[str, Any],
        env_state: dict[str, Any],
    ) -> dict[str, Any]:
        tool_result = env_state.get("tool_result", {})
        tool_name = tool_schema.get("name", "query_db")

        user_content = f"""The user asked: "{query}"

The tool "{tool_name}" has already been called and returned:
{json.dumps(tool_result, indent=2)}

Report your findings to the user based strictly on the result above."""

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_L3_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_content),
        ]

        # No tools offered — evaluating synthesis, not tool call construction
        response = await llm.achat(messages=messages)
        response_text = response.message.content or ""

        if verbose:
            print(f"\n[l3-agent] query:    {query}")
            print(f"[l3-agent] response: {response_text[:300]}")

        return {
            "agent_response": response_text,
            "tool_result": tool_result,
        }

    return agent_fn
