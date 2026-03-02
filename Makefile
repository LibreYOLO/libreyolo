UV := uv run --no-sync

help:
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "                         LibreYOLO Makefile"
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Development Commands:"
	@echo "  setup                         - Create venv and install package + dev dependencies"
	@echo "  check_format                  - Check code formatting"
	@echo "  format                        - Format code with ruff"
	@echo "  lint                          - Run linter"
	@echo "  typecheck                     - Run type checker (ty)"
	@echo "  test                          - Run fast unit tests (no weights needed)"
	@echo "  test_integration              - Run integration tests (needs real model weights)"
	@echo "  test_e2e                      - Run e2e export tests (needs GPU + model weights)"
	@echo "  build                         - Build package"
	@echo "  clean                         - Remove build and test cache artifacts"

# Development Commands
setup:
	uv sync --dev
	@echo ""
	@echo "✅ Setup complete! To activate the virtual environment, run:"
	@echo "   source .venv/bin/activate"

check_format:
	$(UV) ruff format --check

format:
	$(UV) ruff format

lint:
	$(UV) ruff check

typecheck:
	$(UV) ty check

test:
	$(UV) pytest

test_integration:
	$(UV) pytest -m integration

test_e2e:
	@echo "🧹 Cleaning pytest cache before tests..."
	@rm -rf /tmp/pytest-of-$(USER) 2>/dev/null || true
	$(UV) pytest tests/e2e/ -m e2e -v --ignore=tests/e2e/test_rf5_training.py

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