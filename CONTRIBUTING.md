# Contributing to Chatuskoti Evals

Thank you for your interest! This project is a research demonstration of representation sufficiency for research-loop decisions. Contributions should stay scoped to that core thesis.

## Getting Started

1. **Fork** the repository
2. **Create a branch** (`git checkout -b feature/my-feature`)
3. **Install dev dependencies:** `pip install -e ".[dev]"`
4. **Set up pre-commit checks** (optional but recommended):

## Development Workflow

### Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check chatuskoti_evals/ tests/
ruff format chatuskoti_evals/ tests/
```

Type checking with [Mypy](https://mypy-lang.org/):

```bash
mypy chatuskoti_evals/ tests/
```

Both run in CI against Python 3.10–3.12.

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v --tb=short

# Run specific test file
python -m pytest tests/test_scoring.py -v

# Run with coverage
python -m pytest tests/ --cov=chatuskoti_evals
```

### Verification

Before submitting a PR, run the release verification script:

```bash
python scripts/verify_release_bundle.py
```

This checks that tracked artifact bundles are well-formed.

## Pull Request Process

1. **Open an issue first** for significant changes to discuss design.
2. Keep changes **scoped to the representation framework** — this project maintains a narrow claim.
3. Ensure all **tests pass** and **ruff/mypy are clean**.
4. Update **documentation** if adding or changing behavior.
5. Add or update **tests** for any new functionality.
6. The PR will be reviewed for correctness, scope, and clarity.

## What This Project Accepts

- Bug fixes and correctness improvements
- New canonical failure cases that illustrate representational collapse
- Improvements to documentation, tests, or code quality
- Additional backends (datasets/models) that preserve the Vec3 framework
- Trajectory prediction or analysis improvements

## What This Project Typically Declines

- Features outside the representation-sufficiency thesis
- Large-scale benchmark competitions
- Learned or domain-general research controllers
- Changes that would make the project a "general evaluation framework"

## Security

Please see [SECURITY.md](SECURITY.md) for our vulnerability reporting process.

## Code of Conduct

All contributors must abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
