.PHONY: help setup test lint format hooks check

help:
	@echo "make setup    install dependencies and git hooks - run once after cloning"
	@echo "make test     run the tests"
	@echo "make lint     report style problems and likely bugs; change nothing"
	@echo "make format   fix everything that can be fixed automatically"
	@echo "make hooks    run every pre-commit hook on every file"
	@echo "make check    lint, then test - run this before every push"
	@echo "make api      run the API on this Mac, reloading when code changes"
	@echo "make image    build the Docker image"
	@echo "make serve    build the image, then run the API in a container on port 8000"

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

api:
	uv run uvicorn support_assistant.api:app --reload --port 8000

image:
	docker build -t support-assistant .

serve: image
	docker run --rm -p 8000:8000 --name support-assistant support-assistant

check: lint test
