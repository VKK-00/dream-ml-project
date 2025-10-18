# Contributing

Thanks for your interest in improving this project!

## How to contribute

1. **Fork** the repository and create a new branch:
   ```bash
   git checkout -b feat/your-feature
   ```
2. **Install** developer dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pre-commit pytest
   pre-commit install
   ```
3. **Run checks** locally before committing:
   ```bash
   pre-commit run --all-files
   pytest -q
   ```
4. **Open a PR** with a clear description of the changes and rationale.

## Code style

- Lint: `ruff`
- Format: `black`
- Notebooks: outputs are stripped by `nbstripout`
- Keep lines ≤ 100 chars where practical.

## Tests

- Add unit tests for new utilities in `tests/`.
- Keep tests fast and deterministic.

## Data

- Do **not** commit production Excel files. Use small synthetic samples (e.g. `sample_minimal.xlsx`) or document how to acquire data.

## CI

GitHub Actions runs pre-commit and tests on pushes and PRs. Fix any reported issues before merging.

## License

By contributing, you agree your contributions are licensed under the repository’s MIT License.
