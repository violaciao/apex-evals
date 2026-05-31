"""
APEX Eval Configuration
-----------------------
APEX_PROFILE=free        → free backends (Groq Llama-3.1-8b + local stubs)  [default]
APEX_PROFILE=anthropic   → Anthropic Claude Sonnet 4 + real tools
APEX_PROFILE=openai      → OpenAI GPT-4o + real tools
APEX_PROFILE=gemini      → Google Gemini 2.5 Pro + real tools
APEX_PROFILE=mistral     → Mistral Large + real tools
APEX_PROFILE=standard    → alias for anthropic (backwards compat)
"""

from __future__ import annotations
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is installed in normal envs
    def load_dotenv(*args, **kwargs):
        return False


load_dotenv()

PROFILE = os.getenv("APEX_PROFILE", "free")  # "standard" | "free"


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key_env: str
    max_tokens: int = 1024
    temperature: float = 0.0  # deterministic for evals


@dataclass(frozen=True)
class ToolConfig:
    db_url: str          # SQLAlchemy DSN
    use_testcontainers: bool
    use_mcp_stubs: bool


@dataclass(frozen=True)
class ApexConfig:
    profile: str
    llm: LLMConfig
    tools: ToolConfig
    record_mode: str     # vcrpy mode: "none" | "new_episodes" | "all"


_STANDARD_TOOLS = ToolConfig(
    db_url=os.getenv("STANDARD_DB_URL", "postgresql+asyncpg://localhost/apex"),
    use_testcontainers=True,
    use_mcp_stubs=False,
)

_CONFIGS: dict[str, ApexConfig] = {
    "free": ApexConfig(
        profile="free",
        llm=LLMConfig(
            provider="groq",
            model="llama-3.1-8b-instant",
            api_key_env="GROQ_API_KEY",
            max_tokens=1024,
        ),
        tools=ToolConfig(
            db_url="sqlite+aiosqlite:///./fixtures/apex_test.db",
            use_testcontainers=False,
            use_mcp_stubs=True,
        ),
        record_mode="none",  # replay from cassettes; no live calls
    ),
    "anthropic": ApexConfig(
        profile="anthropic",
        llm=LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key_env="ANTHROPIC_API_KEY",
        ),
        tools=_STANDARD_TOOLS,
        record_mode="new_episodes",
    ),
    "openai": ApexConfig(
        profile="openai",
        llm=LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env="OPENAI_API_KEY",
        ),
        tools=_STANDARD_TOOLS,
        record_mode="new_episodes",
    ),
    "gemini": ApexConfig(
        profile="gemini",
        llm=LLMConfig(
            provider="gemini",
            model="gemini-2.5-pro",
            api_key_env="GOOGLE_API_KEY",
        ),
        tools=_STANDARD_TOOLS,
        record_mode="new_episodes",
    ),
    "mistral": ApexConfig(
        profile="mistral",
        llm=LLMConfig(
            provider="mistral",
            model="mistral-large-latest",
            api_key_env="MISTRAL_API_KEY",
        ),
        tools=_STANDARD_TOOLS,
        record_mode="new_episodes",
    ),
}

# backwards compat alias
_CONFIGS["standard"] = _CONFIGS["anthropic"]


def get_config() -> ApexConfig:
    if PROFILE not in _CONFIGS:
        raise ValueError(f"Unknown APEX_PROFILE={PROFILE!r}. Choose: {list(_CONFIGS)}")
    return _CONFIGS[PROFILE]


def get_llm():
    """Return a LlamaIndex LLM instance for the active profile."""
    cfg = get_config().llm
    api_key = os.getenv(cfg.api_key_env, "")

    if cfg.provider == "groq":
        from llama_index.llms.groq import Groq
        return Groq(model=cfg.model, api_key=api_key, max_tokens=cfg.max_tokens)

    if cfg.provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic
        return Anthropic(model=cfg.model, api_key=api_key, max_tokens=cfg.max_tokens)

    if cfg.provider == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(model=cfg.model, api_key=api_key, max_tokens=cfg.max_tokens)

    if cfg.provider == "gemini":
        from llama_index.llms.gemini import Gemini
        return Gemini(model=cfg.model, api_key=api_key, max_tokens=cfg.max_tokens)

    if cfg.provider == "mistral":
        from llama_index.llms.mistralai import MistralAI
        return MistralAI(model=cfg.model, api_key=api_key, max_tokens=cfg.max_tokens)

    raise ValueError(f"Unsupported provider: {cfg.provider}")


CONFIG = get_config()
