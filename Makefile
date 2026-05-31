UV := uv run --no-sync

.DEFAULT_GOAL := help
.PHONY: help setup format lint typecheck test test_install_smoke test_e2e print_nightly_suite test_general_nightly test_flagship_nightly test_nightly test_rf5 build clean

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
	@echo "  test_install_smoke            - Run clean install smoke (MODE=editable|wheel|sdist|pypi)"
	@echo "  test_e2e                      - Run all e2e tests (needs GPU + model weights)"
	@echo "  test_e2e FROM=<file>          - Resume from a test file (e.g. FROM=test_rf1_training.py or FROM=rf1_training)"
	@echo "  test_e2e MARKERS='<expr>'     - Run only matching e2e markers (e.g. MARKERS='e2e and not experimental_backend')"
	@echo "  test_e2e MARKER='<expr>'      - Alias for MARKERS=..., also works with FROM=..."
	@echo "  print_nightly_suite           - Print nightly suite version and contract"
	@echo "  test_general_nightly          - Run broad native inference nightly checks"
	@echo "  test_flagship_nightly         - Run heavy YOLO9/RF-DETR nightly checks"
	@echo "  test_nightly                  - Run general + flagship nightly checks"
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

test_install_smoke:
	$(UV) python tests/smoke/run_install_smoke.py --mode $${MODE:-editable}

test_e2e:
	@if [ -z "$(FROM)" ]; then $(MAKE) clean; fi
	@files=$$(find tests/e2e -name 'test_*.py' | sort); \
	total=$$(echo "$$files" | wc -w); \
	markers="$(MARKERS)"; \
	if [ -z "$$markers" ]; then markers="$(MARKER)"; fi; \
	if [ -z "$$markers" ]; then markers="e2e and not rf5"; fi; \
	resume_from="$(FROM)"; \
	resume_name=""; \
	if [ -n "$$resume_from" ]; then \
		resume_name=$$(basename "$$resume_from"); \
		resume_name=$${resume_name%.py}; \
		case "$$resume_name" in \
			test_*) ;; \
			*) resume_name="test_$$resume_name" ;; \
		esac; \
	fi; \
	i=0; passed=0; failed=0; skipped=0; resuming=0; found_resume=0; \
	if [ -n "$$resume_name" ]; then resuming=1; fi; \
	echo ""; \
	echo "══════════════════════════════════════════════════════════════"; \
	if [ -n "$$resume_name" ]; then \
		echo "  e2e test suite — $$total files (resuming from $$resume_name)"; \
	else \
		echo "  e2e test suite — $$total files (each in its own process)"; \
	fi; \
	echo "  markers: $$markers"; \
	echo "══════════════════════════════════════════════════════════════"; \
	echo ""; \
	for f in $$files; do \
		i=$$((i + 1)); \
		name=$$(basename "$$f" .py); \
		if [ $$resuming -eq 1 ]; then \
			if [ "$$name" = "$$resume_name" ]; then \
				found_resume=1; \
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
		$(UV) pytest "$$f" -m "$$markers" -v; \
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
	if [ -n "$$resume_name" ] && [ $$found_resume -eq 0 ]; then \
		echo ""; \
		echo "  FAILED: resume target '$$resume_from' not found"; \
		echo ""; \
		exit 2; \
	fi; \
	echo ""; \
	echo "══════════════════════════════════════════════════════════════"; \
	echo "  all done: $$passed passed, $$skipped skipped, $$failed failed"; \
	echo "══════════════════════════════════════════════════════════════"

print_nightly_suite:
	@$(UV) python -c "from tests.e2e.nightly_contract import nightly_summary_line; print(nightly_summary_line())"

test_general_nightly: print_nightly_suite
	LIBREYOLO_FAIL_ON_NIGHTLY_SKIP=1 $(MAKE) test_e2e MARKERS='general_nightly'

test_flagship_nightly: print_nightly_suite
	LIBREYOLO_FAIL_ON_NIGHTLY_SKIP=1 $(MAKE) test_e2e MARKERS='flagship_nightly and not export_backend'

test_nightly: print_nightly_suite
	LIBREYOLO_FAIL_ON_NIGHTLY_SKIP=1 $(MAKE) test_e2e MARKERS='general_nightly'
	LIBREYOLO_FAIL_ON_NIGHTLY_SKIP=1 $(MAKE) test_e2e MARKERS='flagship_nightly and not export_backend'

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
