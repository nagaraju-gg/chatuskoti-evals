from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from chatuskoti_evals.models import RunScore, Vec3


class WisdomStore:
    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._truthness: dict[str, float] = defaultdict(float)
        self._reliability: dict[str, float] = defaultdict(float)
        self._validity: dict[str, float] = defaultdict(float)

    def update(self, family: str, run_score: RunScore) -> None:
        n = self._counts[family] + 1
        self._truthness[family] = running_mean(self._truthness[family], run_score.mean.truthness, n)
        self._reliability[family] = running_mean(self._reliability[family], run_score.mean.reliability, n)
        self._validity[family] = running_mean(self._validity[family], run_score.mean.validity, n)
        self._counts[family] = n

    def predict(self, family: str) -> Vec3:
        return Vec3(
            truthness=self._truthness[family],
            reliability=self._reliability[family],
            validity=self._validity[family],
        )

    def family_score(self, family: str) -> float:
        vec = self.predict(family)
        return vec.truthness + 0.35 * vec.reliability + 0.20 * vec.validity

    def confident_families(self, min_seen: int = 2) -> list[str]:
        return sorted(family for family, count in self._counts.items() if count >= min_seen)

    def save(self, path: Path) -> None:
        payload = {
            "counts": dict(self._counts),
            "truthness": dict(self._truthness),
            "reliability": dict(self._reliability),
            "validity": dict(self._validity),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "WisdomStore":
        store = cls()
        if not path.exists():
            return store
        payload = json.loads(path.read_text(encoding="utf-8"))
        for family, count in payload.get("counts", {}).items():
            store._counts[family] = int(count)
        for family, value in payload.get("truthness", {}).items():
            store._truthness[family] = float(value)
        for family, value in payload.get("reliability", payload.get("coherence", {})).items():
            store._reliability[family] = float(value)
        for family, value in payload.get("validity", payload.get("comparability", {})).items():
            store._validity[family] = float(value)
        return store

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            family: {
                "count": self._counts[family],
                "truthness": round(self._truthness[family], 5),
                "reliability": round(self._reliability[family], 5),
                "validity": round(self._validity[family], 5),
                "score": round(self.family_score(family), 5),
            }
            for family in sorted(self._counts)
        }


def running_mean(previous: float, value: float, n: int) -> float:
    return previous + (value - previous) / n
