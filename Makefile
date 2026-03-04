UV := uv run --no-sync

# Reusable pytest cache cleanup (avoids stale state between GPU-heavy test runs)
define PYTEST_CLEANUP
	@echo "🧹 Cleaning pytest cache before tests..."
	@rm -rf /tmp/pytest-of-$(USER) 2>/dev/null || true
endef

.DEFAULT_GOAL := help
.PHONY: help setup format lint typecheck test test_integration test_e2e test_rf1 test_rf5 build clean

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
	@echo "  test_integration              - Run integration tests (needs real model weights)"
	@echo "  test_e2e                      - Run e2e export tests (needs GPU + model weights)"
	@echo "  test_rf1                      - Run RF1 training tests (all 15 models)"
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

test_integration:
	$(UV) pytest -m integration

test_e2e:
	$(PYTEST_CLEANUP)
	$(UV) pytest tests/e2e/ -m "e2e and not rf1 and not rf5" -v

test_rf1:
	$(PYTEST_CLEANUP)
	$(UV) pytest tests/e2e/test_rf1_training.py -m rf1 -v

test_rf5:
	$(PYTEST_CLEANUP)
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