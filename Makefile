# Workshop maintenance tasks. Students do not need these; instructors do.

.PHONY: help install test notebooks check lint clean models smoke

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install core + dev requirements
	pip install -r requirements.txt -r requirements-dev.txt

test:  ## Run the test suite (no model or network needed)
	python -m pytest

notebooks:  ## Rebuild notebooks/*.ipynb from notebook_src/*.py
	python tools/nbbuild.py

check:  ## Verify notebooks are built and valid, links resolve, tests pass
	python tools/nbbuild.py --check
	python tools/check_notebooks.py
	python tools/check_links.py
	python -m pytest

lint:  ## Lint the package, scripts and tools
	ruff check src tests tools scripts
	ruff format --check src tests tools scripts

models:  ## Pre-download the core models
	python scripts/download_models.py --set core

smoke:  ## End-to-end check (loads a real model)
	python scripts/smoke_test.py

clean:  ## Remove caches and generated artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache data/.index outputs
	@echo "Model cache is untouched. Use 'huggingface-cli delete-cache' for that."
