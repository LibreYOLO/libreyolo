UV := uv run --no-sync

.DEFAULT_GOAL := help
.PHONY: help setup format lint typecheck test test_e2e test_rf5 build clean

help:
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "                         LibreYOLO Makefile"
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Development Commands:"
	@echo "  setup                         - Create venv and install package + dev dependencies"
	@echo "  format                        - Format code with ruff"
	@echo "  lint                          - Run linter"
	@echo "  typecheck                     - Run type checker"
	@echo "  test                          - Run fast unit tests (no weights needed)"
	@echo "  test_e2e                      - Run all e2e tests (needs GPU + model weights)"
	@echo "  test_e2e FROM=<name>          - Resume from a specific test file (e.g. FROM=rfdetr_seg)"
	@echo "  test_rf5                      - Run RF5 training benchmark tests"
	@echo "  build                         - Build package"
	@echo "  clean                         - Remove build and test cache artifacts"

setup:
	uv sync --dev
	@echo ""
	@echo "✅ Setup complete! To activate the virtual environment, run:"
	@echo "   source .venv/bin/activate"

format:
	$(UV) ruff format

lint:
	$(UV) ruff check --fix

typecheck:
	$(UV) ty check

test:
	$(UV) pytest

test_e2e:
	@if [ -z "$(FROM)" ]; then $(MAKE) clean; fi
	@files=$$(ls tests/e2e/test_*.py); \
	total=$$(echo "$$files" | wc -w); \
	resume_from="$(FROM)"; \
	i=0; passed=0; failed=0; skipped=0; resuming=0; \
	if [ -n "$$resume_from" ]; then resuming=1; fi; \
	echo ""; \
	echo "══════════════════════════════════════════════════════════════"; \
	if [ -n "$$resume_from" ]; then \
		echo "  e2e test suite — $$total files (resuming from $$resume_from)"; \
	else \
		echo "  e2e test suite — $$total files (each in its own process)"; \
	fi; \
	echo "══════════════════════════════════════════════════════════════"; \
	echo ""; \
	for f in $$files; do \
		i=$$((i + 1)); \
		name=$$(basename "$$f" .py); \
		if [ $$resuming -eq 1 ]; then \
			if echo "$$name" | grep -q "$$resume_from"; then \
				resuming=0; \
			else \
				echo "  [$$i/$$total] $$name — skipped (resuming)"; \
				skipped=$$((skipped + 1)); \
				continue; \
			fi; \
		fi; \
		echo "────────────────────────────────────────────────────────────"; \
		echo "  [$$i/$$total] $$name"; \
		echo "────────────────────────────────────────────────────────────"; \
		$(UV) pytest "$$f" -m "e2e and not rf5" -v; \
		rc=$$?; \
		if [ $$rc -eq 0 ]; then passed=$$((passed + 1)); \
		elif [ $$rc -eq 5 ]; then skipped=$$((skipped + 1)); \
		else failed=$$((failed + 1)); \
			echo ""; \
			echo "  FAILED: $$name (exit $$rc)"; \
			echo ""; \
			exit $$rc; \
		fi; \
	done; \
	echo ""; \
	echo "══════════════════════════════════════════════════════════════"; \
	echo "  all done: $$passed passed, $$skipped skipped, $$failed failed"; \
	echo "══════════════════════════════════════════════════════════════"

test_rf5: clean
	$(UV) pytest tests/e2e/test_rf5_training.py -m rf5 -v

build:
	@echo "📦 Building package..."
	@mkdir -p dist
	uv build --out-dir dist/
	@echo "✅ Package built:"
	@ls -lh dist/*.whl

clean:
	@echo "🧹 Cleaning build and test cache artifacts..."
	@rm -rf dist *.egg-info .ruff_cache .pytest_cache
	@rm -rf /tmp/pytest-of-$(USER) 2>/dev/null || true
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@echo "✅ Clean complete!"