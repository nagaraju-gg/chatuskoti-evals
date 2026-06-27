from __future__ import annotations

import json
import sys
from pathlib import Path

ACTION_COLORS = {
    "adopt": "#2563eb",
    "hold": "#f59e0b",
    "reject": "#dc2626",
    "rollback": "#7c3aed",
    "reframe": "#059669",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 scripts/generate_failure_figure.py <failure_results.json>")
        return 1

    source = Path(sys.argv[1])
    results = json.loads(source.read_text(encoding="utf-8"))
    out_dir = source.parent
    write_svg(out_dir / "benchmark_figure.svg", results)
    write_caption(out_dir / "benchmark_figure_caption.md", results)
    print(f"Wrote figure assets to {out_dir}")
    return 0


def write_svg(path: Path, results: list[dict]) -> None:
    width = 1160
    height = 460
    left = 34

    row_h = 86
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="26" font-family="Helvetica" font-size="22" font-weight="700">'
        'Canonical Failure Cases — Vec3 Resolution</text>',
        f'<text x="{left}" y="50" font-family="Helvetica" font-size="12" fill="#444">'
        'Each case shares T &gt; 0 with clean gains but requires a distinct action</text>',
        '<line x1="34" y1="68" x2="1124" y2="68" stroke="#d1d5db"/>',
        '<text x="42" y="92" font-family="Helvetica" font-size="12" font-weight="700">Case</text>',
        '<text x="282" y="92" font-family="Helvetica" font-size="12" font-weight="700">Metric</text>',
        '<text x="392" y="92" font-family="Helvetica" font-size="12" font-weight="700">Vec3 Action</text>',
        '<text x="532" y="92" font-family="Helvetica" font-size="12" font-weight="700">Key Signals</text>',
        '<text x="950" y="92" font-family="Helvetica" font-size="12" font-weight="700">Why T Alone Is Insufficient</text>',
    ]

    for index, item in enumerate(results):
        y = 104 + index * row_h
        case = simplify_name(item["scenario_name"])
        metric = item["candidate_metric"]
        vec3_action = item["resolution"]["action"]
        signals = ", ".join(item["run_score"]["fired_signals"]) or "none"
        interpretation = interpretation_text(vec3_action)

        lines.extend(
            [
                f'<rect x="34" y="{y-18}" width="1090" height="64" fill={"#f8fafc" if index % 2 == 0 else "#ffffff"} stroke="#e5e7eb"/>',
                f'<text x="42" y="{y}" font-family="Helvetica" font-size="13" font-weight="700">{escape(case)}</text>',
                f'<text x="42" y="{y+18}" font-family="Helvetica" font-size="11" fill="#555">{escape(item["action_spec"]["name"])}</text>',
                f'<text x="282" y="{y}" font-family="Helvetica" font-size="13">{metric:.4f}</text>',
                action_badge(392, y - 14, vec3_action),
                f'<text x="532" y="{y}" font-family="Helvetica" font-size="12">{escape(signals)}</text>',
                f'<text x="950" y="{y}" font-family="Helvetica" font-size="12">{escape(interpretation)}</text>',
            ]
        )

    lines.extend(
        [
            '<line x1="34" y1="426" x2="1124" y2="426" stroke="#d1d5db"/>',
            '<text x="34" y="446" font-family="Helvetica" font-size="12" fill="#444">'
            'All four cases share positive truthness but require different actions — '
            'scalar evaluation cannot distinguish them.</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def action_badge(x: int, y: int, action: str) -> str:
    color = ACTION_COLORS[action]
    label = action.upper()
    width = max(72, 10 * len(label))
    return (
        f'<rect x="{x}" y="{y}" rx="8" ry="8" width="{width}" height="24" fill="{color}" />'
        f'<text x="{x + 10}" y="{y + 16}" font-family="Helvetica" font-size="12" font-weight="700" fill="#ffffff">{label}</text>'
    )


def simplify_name(name: str) -> str:
    mapping = {
        "pyrrhic_probe_injection": "Pyrrhic Pass",
        "goodhart_probe_injection": "Orthogonal Pass",
        "metric_gaming_probe_injection": "Orthogonal Pass",
        "broken_failure_injection": "Broken Failure",
        "incomparable_eval_injection": "Incomparable Pass",
    }
    return mapping.get(name, name.replace("_", " ").title())


def interpretation_text(vec3_action: str) -> str:
    mapping = {
        "hold": "Metric improved but reliability degraded",
        "reframe": "Metric improved but validity collapsed",
        "rollback": "Metric and internals both damaged",
    }
    return mapping.get(vec3_action, f"Action: {vec3_action}")


def write_caption(path: Path, results: list[dict]) -> None:
    body = [
        "Figure caption:",
        "",
        "Canonical failure cases demonstrating representational insufficiency of scalar evaluation. "
        "Each case produces a positive metric delta (T > 0) yet requires a distinct decision action "
        "because the T/R/V decomposition reveals structurally different outcomes. "
        "A scalar (T-only) evaluator would merge all four into 'accept,' losing the distinctions "
        "required for correct research-loop decisions.",
    ]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
