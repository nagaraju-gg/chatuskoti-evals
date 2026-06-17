from __future__ import annotations

import unittest

from chatuskoti_evals.core.wisdom import WisdomStore
from chatuskoti_evals.evaluation.proposals import ProposalEngine


class ProposalTests(unittest.TestCase):
    def test_first_proposal_is_stochastic_depth_high(self) -> None:
        engine = ProposalEngine()
        wisdom = WisdomStore()
        first = engine.propose("vec3", [], wisdom)
        self.assertEqual(first.name, "stochastic_depth_high")

    def test_second_proposal_depends_on_last_signals(self) -> None:
        from chatuskoti_evals.core.models import ActionSpec, HistoryEntry, RunScore, Vec3

        engine = ProposalEngine()
        wisdom = WisdomStore()
        entry = HistoryEntry(
            iteration=1,
            timestamp="",
            controller="vec3",
            action_spec=ActionSpec(name="stochastic_depth_high", family="regularization", params={}, rationale=""),
            baseline_id="b1",
            run_ids=["r1"],
            run_score=RunScore(
                mean=Vec3(truthness=0.1, reliability=-0.2, validity=0.3),
                std=Vec3(truthness=0.0, reliability=0.0, validity=0.0),
                mag=1.0,
                spread=0.0,
                fired_signals=["instability_gap"],
                raw_detectors={},
                axis_components={},
            ),
            resolver_action="reject",
            resolver_reason="test",
            depth=1.0,
            width=1,
            accepted_primary_metric=None,
        )
        second = engine.propose("vec3", [entry], wisdom)
        self.assertEqual(second.name, "cosine_warmup")
