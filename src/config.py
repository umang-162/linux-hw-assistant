"""Runtime configuration.

Settings come from (highest priority first):
  1. CLI flags (passed into ``load_settings``)
  2. Environment variables / a ``.env`` file in the project root
  3. The defaults below

Only two LLM back-ends exist, both free, both spoken to through the
OpenAI-compatible ``openai`` SDK:

* ``groq``   – Groq's free cloud API (default)
* ``ollama`` – a local Ollama server (``--local``), fully offline and private
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Defaults --------------------------------------------------------------------
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Groq retired llama-3.3-70b-versatile on 2026-08-16 (free/dev tiers); its
# documented replacement is openai/gpt-oss-120b. Override with GROQ_MODEL.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"

MOCK_SCENARIOS = ("laptop", "laptop-nosudo", "wsl2", "minimal")


class ConfigError(RuntimeError):
    """Raised when the configuration cannot work (e.g. missing Groq key)."""


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    provider: str                      # "groq" | "ollama"
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.1
    max_tool_rounds: int = 6
    command_timeout: int = 5
    history_char_budget: int = 24000
    llm_timeout: float = 60.0
    physical_ram_slots: int | None = None
    mock: bool = False
    mock_scenario: str = "laptop"
    debug: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return self.provider == "ollama"

    @property
    def provider_label(self) -> str:
        if self.is_local:
            return f"Ollama (local, offline) · {self.model}"
        return f"Groq free cloud API · {self.model}"

    @property
    def data_source_label(self) -> str:
        if self.mock:
            return f"MOCK fixtures (scenario: {self.mock_scenario})"
        return "LIVE system commands on this machine"


def load_settings(
    *,
    local: bool = False,
    mock: bool = False,
    mock_scenario: str | None = None,
    debug: bool = False,
    model_override: str | None = None,
    require_llm: bool = True,
    env_file: Path | None = None,
) -> Settings:
    """Build :class:`Settings` from flags + environment.

    ``require_llm=False`` skips the API-key check, which lets ``--inspect``
    (tool output only, no LLM) work with zero configuration.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    scenario = (mock_scenario or _env_str("MOCK_SCENARIO", "laptop")).lower()
    if scenario not in MOCK_SCENARIOS:
        raise ConfigError(
            f"Unknown mock scenario {scenario!r}. Choose one of: {', '.join(MOCK_SCENARIOS)}"
        )

    common = dict(
        temperature=_env_float("LLM_TEMPERATURE", 0.1, 0.0, 1.0),
        max_tool_rounds=_env_int("MAX_TOOL_ROUNDS", 6, 1, 15),
        command_timeout=_env_int("COMMAND_TIMEOUT", 5, 1, 30),
        mock=mock,
        mock_scenario=scenario,
        debug=debug,
        physical_ram_slots=_env_int("PHYSICAL_RAM_SLOTS", 0, 0, 16) or None,
    )

    if local:
        return Settings(
            provider="ollama",
            base_url=_env_str("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            api_key=_env_str("OLLAMA_API_KEY", "ollama"),
            model=model_override or _env_str("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            history_char_budget=_env_int("HISTORY_CHAR_BUDGET_LOCAL", 10000, 2000, 200000),
            # Small models on old CPUs can need minutes for the first answer.
            llm_timeout=_env_float("LLM_TIMEOUT_LOCAL", 300.0, 10.0, 3600.0),
            **common,
        )

    api_key = _env_str("GROQ_API_KEY", "")
    if require_llm and (not api_key or api_key.startswith("gsk_your_")):
        raise ConfigError(
            "GROQ_API_KEY is not set.\n"
            "  • Get a free key (no credit card) at https://console.groq.com/keys\n"
            "  • Put it in .env  (cp .env.example .env)\n"
            "  • Or run fully offline with:  python main.py --local"
        )
    return Settings(
        provider="groq",
        base_url=_env_str("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL),
        api_key=api_key,
        model=model_override or _env_str("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        history_char_budget=_env_int("HISTORY_CHAR_BUDGET_CLOUD", 24000, 2000, 400000),
        llm_timeout=_env_float("LLM_TIMEOUT_CLOUD", 60.0, 10.0, 600.0),
        **common,
    )
