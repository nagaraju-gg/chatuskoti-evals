from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chatuskoti_evals import __version__


EXPECTED_FILES = {
    "canonical_failure": [
        "manifest.json",
        "failure_injection/summary.md",
        "failure_injection/failure_results.json",
    ],
    "challenge_compare": [
        "manifest.json",
        "comparison.md",
        "comparison_summary.json",
        "controller_comparison.svg",
        "challenge_cases.md",
    ],
    "ablations": [
        "manifest.json",
        "summary.md",
        "summary.json",
        "ablation_summary.svg",
    ],
    "calibration": [
        "manifest.json",
        "summary.md",
        "summary.json",
        "threshold_sweep.svg",
        "default/failure_injection/summary.md",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a versioned torch release bundle.")
    parser.add_argument("bundle_root", type=Path)
    args = parser.parse_args()

    bundle_root = args.bundle_root
    failures: list[str] = []

    if not bundle_root.exists():
        failures.append(f"bundle root does not exist: {bundle_root}")
    else:
        for section, relative_paths in EXPECTED_FILES.items():
            section_root = bundle_root / section
            if not section_root.exists():
                failures.append(f"missing section directory: {section_root}")
                continue
            for relative_path in relative_paths:
                target = section_root / relative_path
                if not target.exists():
                    failures.append(f"missing expected artifact: {target}")
            manifest_path = section_root / "manifest.json"
            if manifest_path.exists():
                failures.extend(validate_manifest(manifest_path, section_root))

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1

    print(f"Verified release bundle at {bundle_root}")
    return 0


def validate_manifest(path: Path, section_root: Path) -> list[str]:
    failures: list[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("schema_version") != 2:
        failures.append(f"{path}: expected schema_version=2")
    if payload.get("package_version") != __version__:
        failures.append(f"{path}: expected package_version={__version__}")
    if not payload.get("benchmark_spec_id"):
        failures.append(f"{path}: missing benchmark_spec_id")
    if not payload.get("generated_at"):
        failures.append(f"{path}: missing generated_at")
    if not payload.get("git_commit"):
        failures.append(f"{path}: missing git_commit")

    for relative_path in payload.get("artifact_paths", {}).values():
        if not relative_path:
            continue
        target = section_root / relative_path
        if not target.exists():
            failures.append(f"{path}: artifact_paths entry does not exist: {target}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
