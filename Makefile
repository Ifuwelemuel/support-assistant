.PHONY: help test lint format check

help:
	@echo "make test     run the tests"
	@echo "make lint     report style problems and likely bugs; change nothing"
	@echo "make format   fix everything that can be fixed automatically"
	@echo "make check    lint, then test - run this before every push"

test:
	uv run pytest -q

lint:
	uv run ruff check
	uv run ruff format --check

format:
	uv run ruff check --fix
	uv run ruff format

check: lint test
