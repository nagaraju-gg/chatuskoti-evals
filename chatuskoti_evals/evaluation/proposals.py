from __future__ import annotations

from chatuskoti_evals.core.models import ActionSpec, HistoryEntry
from chatuskoti_evals.core.wisdom import WisdomStore
from chatuskoti_evals.evaluation.actions import ACTION_LIBRARY


class ProposalEngine:
    """Selects the next action based on controller mode, history, and wisdom store scores."""
    def propose(self, controller: str, history: list[HistoryEntry], wisdom: WisdomStore, mode: str = "default") -> ActionSpec:
        """Choose next action: priority rules first (vec3 recovery, binary fallback), then best by wisdom."""
        tried = {entry.action_spec.name for entry in history}
        last = history[-1] if history else None
        last_signals = set(last.run_score.fired_signals if last else [])

        if not history:
            return self._lookup("stochastic_depth_high")

        if controller == "vec3":
            if {"hyper_coherence", "proxy_decoupling", "instability_gap"} & last_signals and "cosine_warmup" not in tried:
                return self._lookup("cosine_warmup")
            if self._was_adopted(history, "cosine_warmup") and "stochastic_depth_low" not in tried:
                return self._lookup("stochastic_depth_low")
            if "eval_regime_changed" in last_signals and "label_smoothing" not in tried:
                return self._lookup("label_smoothing")
        else:
            if self._was_adopted(history, "stochastic_depth_high") and "stochastic_depth_low" not in tried:
                return self._lookup("stochastic_depth_low")
            if self._was_adopted(history, "stochastic_depth_low") and "focal_objective" not in tried:
                return self._lookup("focal_objective")

        if "adamw" not in tried:
            return self._lookup("adamw")
        if "mixup" not in tried:
            return self._lookup("mixup")
        if "label_smoothing" not in tried:
            return self._lookup("label_smoothing")

        return self._best_by_wisdom(tried, wisdom)

    def _best_by_wisdom(self, tried: set[str], wisdom: WisdomStore) -> ActionSpec:
        remaining = [action for action in ACTION_LIBRARY if action.name not in tried]
        if not remaining:
            return ACTION_LIBRARY[-1]
        return max(remaining, key=lambda action: wisdom.family_score(action.family))

    @staticmethod
    def _lookup(name: str) -> ActionSpec:
        for action in ACTION_LIBRARY:
            if action.name == name:
                return action
        raise KeyError(name)

    @staticmethod
    def _was_adopted(history: list[HistoryEntry], action_name: str) -> bool:
        return any(entry.action_spec.name == action_name and entry.resolver_action == "adopt" for entry in history)
