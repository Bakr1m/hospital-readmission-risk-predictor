.PHONY: install install-train install-dev test lint train serve mlflow-ui docker-build docker-run clean

install:            ## Install serving deps into .venv
	.venv/bin/pip install -r requirements.txt

install-train: install  ## + training deps (xgboost, mlflow)
	.venv/bin/pip install -r requirements-train.txt

install-dev: install  ## + dev tools (pytest, ruff)
	.venv/bin/pip install -r requirements-dev.txt

test:               ## Run the test suite
	.venv/bin/python -m pytest tests/ -q

lint:               ## Lint src/api/tests with ruff
	.venv/bin/ruff check src api tests

train:              ## Retrain all models (logs to MLflow, saves models/readmission_best.joblib)
	.venv/bin/python src/train.py

serve:              ## Run the API locally on :8000
	.venv/bin/python api/main.py

mlflow-ui:          ## Browse experiment runs
	.venv/bin/mlflow ui --backend-store-uri ./mlruns

docker-build:       ## Build the serving image
	docker build -t readmission-api .

docker-run:         ## Run the serving image on :8000
	docker run -p 8000:8000 readmission-api

clean:              ## Remove caches (never touches data/, models/, mlruns/)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
