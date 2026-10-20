.DEFAULT_GOAL := help
.PHONY: help install dev lint format typecheck test cov check eval up down logs clean

help: ## List the available commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

install: ## Install dependencies and the git hooks
	uv sync
	uv run pre-commit install

dev: ## Run the API with reload (in-memory Qdrant, needs .env)
	uv run uvicorn prior_auth.api:app --reload

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format and fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Static types (mypy --strict)
	uv run mypy

test: ## Offline test suite (no network, no API key)
	uv run pytest

cov: ## Tests with a coverage report
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test ## Everything CI runs

eval: ## Evaluate on the labelled cases with the model in .env
	uv run prior-auth-eval --delay 4

up: ## Build and start Qdrant + API (http://localhost:8000)
	docker compose up --build -d --wait

down: ## Stop the stack (volumes are kept)
	docker compose down

logs: ## Follow container logs
	docker compose logs -f

clean: ## Remove caches and local storage
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov storage
