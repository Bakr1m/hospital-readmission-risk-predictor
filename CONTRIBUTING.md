# Contributing

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
make install        # serving deps
make install-train  # + training deps (xgboost, mlflow)
make install-dev    # + dev tools (pytest, ruff)
make test           # run the test suite
```

## Ground rules

1. **No large files in git.** Data (`data/`), MLflow runs (`mlruns/`), and
   regenerable model artifacts stay out of version control — see `.gitignore`.
   The only versioned artifact is the production model
   (`models/readmission_best.joblib`) that the Docker image serves.
2. **Retraining is reproducible.** `python src/train.py` (or `make train`)
   must regenerate `models/readmission_best.joblib` and log to MLflow.
   Pin any new dependency in `requirements*.txt`.
3. **Tests before push.** `make test` must pass. New preprocessing logic needs
   a test in `tests/` that imports the real function (no logic duplication).
   The suite must also pass **without** `data/` present (synthetic fallback),
   because CI checks out the repo with data gitignored.
4. **Lint.** `make lint` (ruff) must be clean.
5. **Small, described commits.** One logical change per commit, imperative
   message (`"Fix ..."`, `"Add ..."`).
6. **Docs with behavior changes.** If `/predict` input/output changes, update
   `README.md`, `example_request.json`, and `docs/model_card.md` in the same PR.
