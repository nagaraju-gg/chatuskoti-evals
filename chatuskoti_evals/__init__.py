"""Benchmark-specific evaluator for research-loop decisions."""

__all__ = ["__version__"]
__version__ = "1.3.0"

_module_redirects = {
    "actions": "chatuskoti_evals.evaluation.actions",
    "benchmark": "chatuskoti_evals.evaluation.benchmark",
    "config": "chatuskoti_evals.core.config",
    "coupling": "chatuskoti_evals.core.coupling",
    "models": "chatuskoti_evals.core.models",
    "progress": "chatuskoti_evals.evaluation.progress",
    "proposals": "chatuskoti_evals.evaluation.proposals",
    "reporting": "chatuskoti_evals.evaluation.reporting",
    "resolver": "chatuskoti_evals.evaluation.resolver",
    "runner": "chatuskoti_evals.evaluation.runner",
    "scenarios": "chatuskoti_evals.evaluation.scenarios",
    "scoring": "chatuskoti_evals.evaluation.scoring",
    "wisdom": "chatuskoti_evals.core.wisdom",
}


def __getattr__(name):
    if name in _module_redirects:
        import importlib

        return importlib.import_module(_module_redirects[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
