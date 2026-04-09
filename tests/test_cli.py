from __future__ import annotations

import unittest

from chatuskoti_evals.cli import build_config, build_parser


class CliTests(unittest.TestCase):
    def test_legacy_ablation_alias_normalizes_to_v1_1_name(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["compare", "--ablation", "no_goodhart"])
        cfg = build_config(args)

        self.assertEqual(cfg.ablation.name, "no_validity")
        self.assertFalse(cfg.detector.enable_validity)
        self.assertTrue(cfg.detector.enable_reliability)

    def test_calibration_command_is_available(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run-calibration"])

        self.assertEqual(args.command, "run-calibration")
