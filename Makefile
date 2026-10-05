# Workshop maintenance tasks. Students do not need these; instructors do.

.PHONY: help install install-locked lock test notebooks diagrams check lint clean models smoke

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install core + dev requirements (latest compatible)
	pip install -r requirements.txt -r requirements-dev.txt

install-locked:  ## Install the exact versions used for teaching
	pip install -r requirements-lock.txt -r requirements-dev.txt

lock:  ## Regenerate requirements-lock.txt from the CURRENT environment
	@echo "Only run this where you have verified the notebooks actually work."
	pip freeze --exclude-editable > requirements-lock.txt
	@echo "Wrote requirements-lock.txt - commit it with a note on what you tested."

test:  ## Run the test suite (no model or network needed)
	python -m pytest

notebooks:  ## Rebuild notebooks/*.ipynb from notebook_src/*.py
	python tools/nbbuild.py

diagrams:  ## Regenerate docs/assets/diagrams/*.svg
	python tools/make_diagrams.py

check:  ## Verify notebooks and diagrams are built and valid, links resolve, tests pass
	python tools/nbbuild.py --check
	python tools/make_diagrams.py --check
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
