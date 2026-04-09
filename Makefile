.PHONY: start-worker execute signal query installdeps

## Send a signal to a running execution
## Usage: make signal id=<execution-id> name=reviewer-decision input='{"approved": true, "comment": "Ship it."}'
signal:
	uv run python src/workflows/interact.py --execution-id $(id) --signal $(name) $(if $(input),--input '$(input)',)

## Query a running execution
## Usage: make query id=<execution-id> name=review-status
query:
	uv run python src/workflows/interact.py --execution-id $(id) --query $(name)

## Install dependencies
installdeps:
	uv sync

## Auto-discover all workflows and start the worker (with file-watch auto-reload)
start-worker:
	uv run python src/dev_worker.py

## Trigger a workflow execution
## Usage: make execute workflow=hello-world input='{"name": "World"}'
execute:
	uv run python src/workflows/start.py $(if $(workflow),--workflow $(workflow),) $(if $(input),--input '$(input)',)
