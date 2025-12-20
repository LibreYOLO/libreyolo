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

build:
	@echo "📦 Building package..."
	@mkdir -p dist
	uv build --out-dir dist/
	@echo "✅ Package built:"
	@ls -lh dist/*.whl

clean:
	@echo "🧹 Cleaning build and test cache artifacts..."
	@rm -rf dist *.egg-info .ruff_cache
	@echo "✅ Clean complete!"