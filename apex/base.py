"""
apex/base.py
------------
Base interface every APEX eval module implements.
All 19 failure modes subclass EvalModule.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time


class DetectionDifficulty(str, Enum):
    LOW = "LOW"              # explicit error thrown
    MEDIUM = "MEDIUM"        # detectable with monitoring
    MEDIUM_HIGH = "MEDIUM_HIGH"
    HIGH = "HIGH"            # silent — no error thrown
    VERY_HIGH = "VERY_HIGH"  # emergent, cross-tool


class FailureLayer(str, Enum):
    L1_TOOL_SELECTION = "L1_TOOL_SELECTION"
    L2_INPUT_CONSTRUCTION = "L2_INPUT_CONSTRUCTION"
    L3_OUTPUT_CONSUMPTION = "L3_OUTPUT_CONSUMPTION"
    L4_CHAIN_MULTITOOL = "L4_CHAIN_MULTITOOL"


@dataclass
class Scenario:
    """A single test case: user intent + environment state."""
    name: str
    user_query: str
    tool_schema: dict[str, Any]         # what tools are available
    env_state: dict[str, Any]           # DB contents, API state, etc.
    expected_tool_call: dict[str, Any]  # what a correct agent would do
    injected_fault: dict[str, Any]      # what makes this scenario fail
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Output of running one scenario through an agent."""
    scenario_name: str
    failure_mode: str
    layer: FailureLayer
    passed: bool
    score: float                        # 0.0 – 1.0
    detection_difficulty: DetectionDifficulty
    actual_tool_call: dict[str, Any] | None
    agent_response: str
    failure_detected: bool              # did the agent catch its own error?
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return (
            f"{status} [{self.layer.value}] {self.failure_mode} "
            f"| score={self.score:.2f} | detected={self.failure_detected} "
            f"| {self.latency_ms:.0f}ms"
        )


class EvalModule(ABC):
    """
    Abstract base for all 19 APEX failure mode evaluators.

    Subclass contract:
        - failure_mode: short identifier e.g. "semantic_argument_error"
        - layer: which of the 4 layers
        - detection_difficulty: from the taxonomy
        - scenarios(): list of Scenario objects
        - run(scenario, agent_fn): execute one scenario, return EvalResult
        - score(result): compute 0–1 score from raw result dict
    """

    failure_mode: str
    layer: FailureLayer
    detection_difficulty: DetectionDifficulty

    @abstractmethod
    def scenarios(self) -> list[Scenario]:
        """Return all test scenarios for this failure mode."""
        ...

    @abstractmethod
    async def run(
        self,
        scenario: Scenario,
        agent_fn: Any,           # callable: (query, tools, env) -> raw_result
    ) -> EvalResult:
        """Execute one scenario against an agent, return EvalResult."""
        ...

    @abstractmethod
    def score(self, raw_result: dict[str, Any]) -> float:
        """Map raw agent output to 0–1 correctness score."""
        ...

    async def run_all(self, agent_fn: Any) -> list[EvalResult]:
        """Convenience: run all scenarios, return results list."""
        results = []
        for scenario in self.scenarios():
            result = await self.run(scenario, agent_fn)
            results.append(result)
        return results

    def _timer(self):
        return time.monotonic()

    def _elapsed_ms(self, start: float) -> float:
        return (time.monotonic() - start) * 1000
