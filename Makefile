# Reproduction pipeline.
#
#   make setup      install pinned dependencies
#   make data       check the committed workbooks are present and intact
#   make notebooks  execute notebooks 01-10 in order
#   make figures    regenerate both figures
#   make verify     check every claim in paper_numbers.yaml
#   make test       run the pytest suite
#   make all        data -> notebooks -> verify
#   make clean      remove generated results (never touches data/)

PYTHON    ?= python3
JUPYTER   ?= jupyter
NOTEBOOKS := $(sort $(wildcard notebooks/*.ipynb))
EXEC_DIR  := results/logs/executed
TIMEOUT   := 1800

.PHONY: setup data notebooks figures verify test all clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/check_data.py --strict

notebooks:
	@mkdir -p $(EXEC_DIR) results/tables results/figures results/logs
	@for nb in $(NOTEBOOKS); do \
		echo "--- executing $$nb"; \
		$(JUPYTER) nbconvert --to notebook --execute \
			--ExecutePreprocessor.timeout=$(TIMEOUT) \
			--output-dir=$(EXEC_DIR) "$$nb" > /dev/null || exit 1; \
	done
	@echo "All notebooks executed. Repository copies remain output-stripped;"
	@echo "executed copies with outputs are under $(EXEC_DIR)/ (git-ignored)."

figures:
	@mkdir -p $(EXEC_DIR) results/figures
	$(JUPYTER) nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=$(TIMEOUT) \
		--output-dir=$(EXEC_DIR) notebooks/09_figures.ipynb > /dev/null
	@ls -l results/figures

verify:
	$(PYTHON) scripts/verify_paper_numbers.py

test:
	$(PYTHON) -m pytest -v

all: data notebooks verify

# Removes regenerable output only. Never touches data/, and never removes a
# tracked file: results/logs/verification.md and results/logs/anonymity_audit.md
# are committed evidence, not intermediate output.
clean:
	rm -rf $(EXEC_DIR)
	rm -f results/tables/*.csv results/figures/fig_* results/logs/comet_qn_results.md
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	@echo "Generated results removed. data/ and tracked logs untouched."
