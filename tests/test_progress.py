from __future__ import annotations

import unittest

from chatuskoti_evals.evaluation.progress import RunProgressContext, RunProgressTracker


class ProgressTrackerTests(unittest.TestCase):
    def test_context_label_includes_iteration_and_action(self) -> None:
        context = RunProgressContext(controller="vec3", phase="iteration", iteration=2, action_name="mixup")
        self.assertEqual(context.label, "vec3:iteration2:mixup")

    def test_tracker_reports_remaining_runs(self) -> None:
        tracker = RunProgressTracker(total_runs=4)
        context = RunProgressContext(controller="binary", phase="baseline")

        first = tracker.start_run(context, seed=0)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.current_run, 1)
        self.assertEqual(first.remaining_runs, 3)

        tracker.finish_run()
        second = tracker.start_run(context, seed=1)
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(second.current_run, 2)
        self.assertEqual(second.remaining_runs, 2)

    def test_adopted_baseline_label_stays_distinct(self) -> None:
        context = RunProgressContext(
            controller="vec3",
            phase="adopted_baseline",
            iteration=1,
            action_name="mixup",
        )
        self.assertEqual(context.label, "vec3:adopted_baseline:iteration1:mixup")
