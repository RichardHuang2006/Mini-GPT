# Mini-GPT — from-scratch GPT pretraining + post-training pipeline
#
# Targets: setup (venv + deps) · test (pytest) · bench · clean · help
#
# `make test` uses whatever `python` is on PATH, so it works against an
# already-installed torch without `make setup` first. `make setup` creates a
# local venv for a clean-room install.

PYTHON ?= python3
VENV    = .venv
VENVBIN = $(VENV)/bin

.PHONY: setup test bench clean help
.DEFAULT_GOAL := help

# ------------------------------------------------------------------- setup ---
# Create a local venv and install pinned dependencies for a fresh clone.
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENVBIN)/pip install --upgrade pip
	$(VENVBIN)/pip install -r requirements.txt
	@echo "setup: activate with 'source $(VENVBIN)/activate'"

# -------------------------------------------------------------------- test ---
# Run the suite. CPU-only sections (config, tokenizer, data) always run; the
# CUDA kernel differential tests are skipped automatically without a GPU.
test:
	$(PYTHON) -m pytest -q

# ------------------------------------------------------------------- bench ---
# Throughput / MFU / peak-VRAM / roofline point.
bench:
	$(PYTHON) -m mini_gpt.bench

# --------------------------------------------------------------- housekeeping
clean:
	rm -rf $(VENV) out .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

help:
	@echo "Mini-GPT targets:"
	@echo "  setup    create .venv and install pinned requirements"
	@echo "  test     run the pytest suite (CUDA kernel tests skip without a GPU)"
	@echo "  bench    tok/s, MFU, peak VRAM, roofline point"
	@echo "  clean    remove .venv, out/, caches, and __pycache__"
