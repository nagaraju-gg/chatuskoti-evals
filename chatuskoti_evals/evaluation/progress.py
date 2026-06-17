from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunProgressContext:
    controller: str
    phase: str
    iteration: int | None = None
    action_name: str | None = None

    @property
    def label(self) -> str:
        if self.phase == "baseline":
            return f"{self.controller}:baseline"
        if self.phase == "iteration":
            if self.iteration is not None and self.action_name:
                return f"{self.controller}:iteration{self.iteration}:{self.action_name}"
            if self.iteration is not None:
                return f"{self.controller}:iteration{self.iteration}"
        parts = [self.controller, self.phase]
        if self.iteration is not None:
            parts.append(f"iteration{self.iteration}")
        if self.action_name:
            parts.append(self.action_name)
        return ":".join(parts)


@dataclass(frozen=True)
class RunProgressSnapshot:
    current_run: int
    total_runs: int
    remaining_runs: int
    seed: int
    context: RunProgressContext
    cache_hit: bool = False


class RunProgressTracker:
    def __init__(self, total_runs: int):
        if total_runs <= 0:
            raise ValueError("total_runs must be positive")
        self.total_runs = total_runs
        self.completed_runs = 0

    def start_run(
        self,
        context: RunProgressContext | None,
        *,
        seed: int,
        cache_hit: bool = False,
    ) -> RunProgressSnapshot | None:
        if context is None:
            return None
        current_run = min(self.completed_runs + 1, self.total_runs)
        remaining_runs = max(self.total_runs - current_run, 0)
        return RunProgressSnapshot(
            current_run=current_run,
            total_runs=self.total_runs,
            remaining_runs=remaining_runs,
            seed=seed,
            context=context,
            cache_hit=cache_hit,
        )

    def finish_run(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self.completed_runs = min(self.completed_runs + count, self.total_runs)

    def add_runs(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self.total_runs += count
