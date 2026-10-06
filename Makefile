.PHONY: help setup test lint format hooks check

help:
	@echo "make setup    install dependencies and git hooks - run once after cloning"
	@echo "make test     run the tests"
	@echo "make lint     report style problems and likely bugs; change nothing"
	@echo "make format   fix everything that can be fixed automatically"
	@echo "make hooks    run every pre-commit hook on every file"
	@echo "make check    lint, then test - run this before every push"

setup:
	uv sync
	uv run pre-commit install

test:
	uv run pytest -q

lint:
	uv run ruff check
	uv run ruff format --check

format:
	uv run ruff check --fix
	uv run ruff format

hooks:
	uv run pre-commit run --all-files

check: lint test
