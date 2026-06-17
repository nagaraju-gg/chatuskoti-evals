from __future__ import annotations

import unittest


class PublicSurfaceTests(unittest.TestCase):
    def test_top_level_exports_are_accessible(self) -> None:
        import chatuskoti_evals  # noqa: F401

    def test_core_modules_importable(self) -> None:
        from chatuskoti_evals import config, coupling, models, wisdom  # noqa: F401

    def test_evaluation_modules_importable(self) -> None:
        from chatuskoti_evals import actions, benchmark, reporting, resolver, runner, scoring  # noqa: F401

    def test_main_module_importable(self) -> None:
        from chatuskoti_evals.__main__ import main  # noqa: F401
