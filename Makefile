# `make verify` is the one command this repository promises a reviewer. It must pass
# from a fresh clone with no network, so every target below reads committed fixtures
# rather than reaching for a model or a server.

PY ?= python3
PKG = semantic_cache_audit
EMBEDDER ?= st:sentence-transformers/all-MiniLM-L6-v2
BUDGET ?= 0.05

.PHONY: help install lint test coverage experiments receipts prove verify \
        corpus embeddings embeddings-live audit report diagrams evidence demo clean

help:
	@echo "install           editable install with dev extras"
	@echo "lint              ruff over src, tests, tools, experiments"
	@echo "test              pytest with junit and coverage reports"
	@echo "experiments       re-run E1 to E5 into docs/experiments"
	@echo "receipts          re-measure every published figure and check the documents"
	@echo "prove             watch the number checker fail on a deliberately broken figure"
	@echo "verify            lint + test + receipts. Offline. The one command that matters."
	@echo "audit             run the headline audit and exit with its code"
	@echo "report            write the HTML report to docs/evidence"
	@echo "corpus            regenerate the adversarial corpus"
	@echo "embeddings        regenerate every offline embedding fixture"
	@echo "embeddings-live   regenerate the pretrained fixture. Needs weights and network."
	@echo "evidence          diagrams, screenshots and the demo recording"

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check src tests tools experiments

test:
	mkdir -p reports
	$(PY) -m pytest -q --junitxml=reports/junit.xml \
		--cov=$(PKG) --cov-report=json:reports/coverage.json --cov-report=term

coverage: test

experiments:
	cd experiments && PYTHONPATH=. $(PY) e1_threshold_curve.py
	cd experiments && PYTHONPATH=. $(PY) e2_class_breakdown.py
	cd experiments && PYTHONPATH=. $(PY) e3_eviction_sweep.py
	cd experiments && PYTHONPATH=. $(PY) e4_budget_decisions.py
	cd experiments && PYTHONPATH=. $(PY) e5_threshold_transfer.py

# Runs the suite and the experiments itself when called directly. CI calls the
# collector with --skip-tests after running the suite as its own step, so a red test
# surfaces as a red test rather than as a traceback from a metrics script.
receipts: experiments
	$(PY) tools/collect_metrics.py --skip-tests
	$(PY) tools/check_numbers.py
	$(PY) tools/check_durations.py

prove:
	$(PY) tools/check_numbers.py --prove

verify: lint test receipts

corpus:
	$(PY) -m $(PKG).corpus.builder

embeddings:
	$(PY) tools/build_fixtures.py

embeddings-live:
	$(PY) tools/build_fixtures.py --pretrained

audit:
	$(PY) -m $(PKG).cli audit --embedder "$(EMBEDDER)" --budget $(BUDGET)

report:
	$(PY) -m $(PKG).cli report --embedder "$(EMBEDDER)" --budget $(BUDGET) \
		--out docs/evidence/report.html

diagrams:
	$(PY) tools/render_diagrams.py

evidence: diagrams report
	$(PY) tools/capture_screenshots.py
	$(PY) tools/record_demo.py

clean:
	rm -rf reports .pytest_cache .ruff_cache .coverage coverage.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
