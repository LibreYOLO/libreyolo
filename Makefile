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
	@echo "  test                          - Run fast unit tests (no weights needed)"
	@echo "  test_integration              - Run integration tests (needs real model weights)"
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

test:
	$(UV) pytest

test_integration:
	$(UV) pytest -m integration

build:
	@echo "📦 Building package..."
	@mkdir -p dist
	uv build --out-dir dist/
	@echo "✅ Package built:"
	@ls -lh dist/*.whl

clean:
	@echo "🧹 Cleaning build and test cache artifacts..."
	@rm -rf dist *.egg-info .ruff_cache .pytest_cache
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@echo "✅ Clean complete!"