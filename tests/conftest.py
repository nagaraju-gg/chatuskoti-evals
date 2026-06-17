from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def run_log_path(temp_output_dir: Path) -> Generator[str, None, None]:
    log_path = temp_output_dir / "logs" / "runs.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("CHATUSKOTI_RUN_LOG_PATH")
    os.environ["CHATUSKOTI_RUN_LOG_PATH"] = str(log_path)
    yield str(log_path)
    if previous is None:
        os.environ.pop("CHATUSKOTI_RUN_LOG_PATH", None)
    else:
        os.environ["CHATUSKOTI_RUN_LOG_PATH"] = previous
