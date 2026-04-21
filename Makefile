UV := uv run --no-sync

.DEFAULT_GOAL := help
.PHONY: help setup format lint typecheck test test_e2e test_rf1 test_rf5 build clean

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
	@echo "  test_e2e                      - Run local e2e tests (no live dataset downloads)"
	@echo "  test_rf1                      - Run RF1 network-backed training tests"
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

test_e2e: clean
	@echo "Running local e2e tests in isolated processes (excludes network-backed suites)..."
	@for f in tests/e2e/test_*.py; do \
		$(UV) pytest "$$f" -m "e2e and not network" -v; \
		rc=$$?; if [ $$rc -ne 0 ] && [ $$rc -ne 5 ]; then exit $$rc; fi; \
	done

test_rf1: clean
	$(UV) pytest tests/e2e/test_rf1_training.py -m "rf1 and network" -v

test_rf5: clean
	$(UV) pytest tests/e2e/test_rf5_training.py -m "rf5 and network" -v

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
