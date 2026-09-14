# DoomWorm developer shortcuts. Everything runs through uv; nothing is installed globally.

.DEFAULT_GOAL := help
UV := uv run
BRAIN ?= runs/small_evolved.json
SEED ?= 1003
NEURON ?= ASHL

.PHONY: help sync hooks test test-fast cov lint format typecheck check clean \
	    demo-0 demo-1 demo-2 demo-3 demo-5 demo-6 demos train play stimulate fetch-data

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- setup -------------------------------------------------------------------

sync: ## Install/refresh the venv from uv.lock (all groups)
	uv sync --all-groups

hooks: sync ## Install pre-commit hooks into .git
	$(UV) pre-commit install

# --- quality -----------------------------------------------------------------

test: ## Run the test suite
	$(UV) pytest

test-fast: ## Run tests, stop at first failure
	$(UV) pytest -x -q

cov: ## Tests with coverage report
	$(UV) pytest --cov --cov-report=term-missing

lint: ## Ruff lint (with autofix) and format check
	$(UV) ruff check . --fix
	$(UV) ruff format --check .

format: ## Ruff format in place
	$(UV) ruff format .

typecheck: ## Strict mypy
	$(UV) mypy

check: lint typecheck test ## Everything CI runs

clean: ## Remove caches and experiment outputs (keeps the venv)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov runs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# --- stage demos -------------------------------------------------------------

demo-0: ## Stage 0: three-neuron chain
	$(UV) python -m doomworm.experiments.three_neurons

demo-1: ## Stage 1: obstacle avoidance (PNG in runs/)
	$(UV) python -m doomworm.experiments.obstacle_agent --plot

demo-2: ## Stage 2: food and hunger (PNG in runs/)
	$(UV) python -m doomworm.experiments.food_agent --plot

demo-3: ## Stage 3: reward comparison of three brains
	$(UV) python -m doomworm.experiments.reward_demo

demo-5: ## Stage 5: connectome statistics
	$(UV) python -m doomworm.experiments.connectome_stats

demo-6: ## Stage 6: debug screen GIF (runs/stage6_debug.gif)
	$(UV) python -m doomworm.experiments.debug_screen_demo

demo-7: ## Stage 7: sensory mapping and per-channel responders
	$(UV) python -m doomworm.experiments.sensory_mapping_demo

demo-8: ## Stage 8: motor mapping, wheel commands per channel
	$(UV) python -m doomworm.experiments.motor_mapping_demo

demos: demo-0 demo-1 demo-2 demo-3 demo-5 demo-6 demo-7 demo-8 ## Run every stage demo

# --- training and connectome -------------------------------------------------

train: ## Stage 4: evolve the small network -> $(BRAIN)
	$(UV) doomworm train --out $(BRAIN)

play: ## Replay $(BRAIN) on seed $(SEED) with a plot
	$(UV) doomworm play --brain $(BRAIN) --seed $(SEED) --plot

stimulate: ## Stage 6: stimulate $(NEURON) and plot propagation
	$(UV) doomworm stimulate $(NEURON) --plot

fetch-data: ## Re-download the Cook 2019 connectome into data/connectome/cook2019
	uv run --with openpyxl scripts/fetch_connectome.py
