# DoomWorm developer shortcuts. Everything runs through uv; nothing is installed globally.

.DEFAULT_GOAL := help
UV := uv run
SEED ?= 1003
NEURON ?= ASHL
LEVEL ?= doom4
STOCK ?= stock_defend
CANDIDATE ?= worm
WORKERS ?= 6
TRAIN_LEVEL ?= doom4
DOOM_SEED ?= 0
WORM_BRAIN ?= docs/brains/a1/worm_evolved_random.json
WATCH_BRAIN ?= docs/brains/doom/doom4/seed0/worm_from_worm_evolved_random.json
WATCH_SEED ?= 3000
DOOM_BENCH = --task doom --sensors ideal --steps 600 --test-seeds 12 --repeats 1

.PHONY: help sync hooks fetch-data test test-fast cov lint format typecheck check clean \
	    demo-0 demo-1 demo-2 demo-3 demo-5 demo-6 demo-7 demo-8 demo-9 demos \
	    demo-b1 demo-b4 demo-b11 watch-doom watch-doomguy watch-stock \
	    benchmark-doom benchmark-vizdoom evolve-doom train-worm compare play stimulate

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- setup -------------------------------------------------------------------

sync: ## Install/refresh the venv from uv.lock (all groups)
	uv sync --all-groups

hooks: sync ## Install pre-commit hooks into .git
	$(UV) pre-commit install

fetch-data: ## Download the connectome dataset
	$(UV) python scripts/fetch_connectome.py

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

# --- the platform, stage by stage --------------------------------------------

demo-0: ## Three neurons in a chain
	$(UV) python -m doomworm.experiments.three_neurons

demo-1: ## Obstacle avoidance (PNG in runs/)
	$(UV) python -m doomworm.experiments.obstacle_agent --plot

demo-2: ## The food task: the curriculum world (PNG in runs/)
	$(UV) python -m doomworm.experiments.food_agent --plot

demo-3: ## Reward comparison of three brains
	$(UV) python -m doomworm.experiments.reward_demo

demo-5: ## Connectome statistics
	$(UV) python -m doomworm.experiments.connectome_stats

demo-6: ## Debug screen GIF (runs/stage6_debug.gif)
	$(UV) python -m doomworm.experiments.debug_screen_demo

demo-7: ## Sensory mapping and per-channel responders
	$(UV) python -m doomworm.experiments.sensory_mapping_demo

demo-8: ## Motor mapping, commands per channel
	$(UV) python -m doomworm.experiments.motor_mapping_demo

demo-9: ## The untrained worm drives the agent (PNG + GIF + JSONL log)
	$(UV) python -m doomworm.experiments.worm_agent --seed $(SEED) --plot --gif

demos: demo-0 demo-1 demo-2 demo-3 demo-5 demo-6 demo-7 demo-8 demo-9 ## Run every platform demo

train-worm: ## The curriculum: evolve the connectome on the food task
	$(UV) doomworm train --scenario worm --maps random

compare: ## Real vs random vs shuffled vs dense topology
	$(UV) doomworm compare

# --- Doom --------------------------------------------------------------------

demo-b1: ## The hand-written floor on a simulator level, then a worm replay
	$(UV) doomworm benchmark --scripted doomguy --maps $(LEVEL) $(DOOM_BENCH) --out-dir runs/benchmark_doom/$(LEVEL)
	$(UV) doomworm play --brain $(WORM_BRAIN) --maps $(LEVEL) --task doom --seed $(SEED) --plot

demo-b4: ## The same level inside the Doom engine (needs the doom group)
	$(UV) doomworm benchmark --scripted doomguy --maps viz$(LEVEL) $(DOOM_BENCH) --out-dir runs/benchmark_doom/floors/eval_viz$(LEVEL)
	$(UV) doomworm play --brain $(WORM_BRAIN) --maps viz$(LEVEL) --task doom --seed $(SEED) --plot

demo-b11: ## A stock ViZDoom scenario ($(STOCK)): the floor and a trained worm
	$(UV) doomworm benchmark --scripted doomguy --maps $(STOCK) --task doom --sensors ideal --steps 600 --test-seeds 4 --repeats 1 --out-dir runs/benchmark_doom/stock
	$(UV) doomworm benchmark --brain $(WATCH_BRAIN) --maps $(STOCK) --task doom --sensors ideal --steps 600 --test-seeds 4 --repeats 1 --out-dir runs/benchmark_doom/stock

evolve-doom: ## Train $(CANDIDATE) on $(LEVEL), seed $(DOOM_SEED) -> runs/doom/
	$(UV) doomworm evolve --candidate $(CANDIDATE) --maps $(LEVEL) --task doom --sensors ideal --steps 600 --workers $(WORKERS) --seed $(DOOM_SEED) --out runs/doom/$(LEVEL)/seed$(DOOM_SEED)/$(CANDIDATE).json

benchmark-doom: ## The floors and every brain of runs/doom/$(TRAIN_LEVEL)/seed$(DOOM_SEED)/ on $(LEVEL)
	$(UV) doomworm benchmark --scripted doomguy --maps $(LEVEL) $(DOOM_BENCH) --out-dir runs/benchmark_doom/floors/eval_$(LEVEL)
	$(UV) doomworm benchmark --scripted follower --maps $(LEVEL) $(DOOM_BENCH) --out-dir runs/benchmark_doom/floors/eval_$(LEVEL)
	for b in $(wildcard runs/doom/$(TRAIN_LEVEL)/seed$(DOOM_SEED)/*.json); do \
	  case $$b in *.meta.json) ;; *) $(UV) doomworm benchmark --brain $$b --maps $(LEVEL) $(DOOM_BENCH) --out-dir runs/benchmark_doom/train_$(TRAIN_LEVEL)/seed$(DOOM_SEED)/eval_$(LEVEL) || exit 1;; esac; done

benchmark-vizdoom: ## The same rows in the Doom engine, level viz$(LEVEL)
	$(UV) doomworm benchmark --scripted doomguy --maps viz$(LEVEL) $(DOOM_BENCH) --out-dir runs/benchmark_doom/floors/eval_viz$(LEVEL)
	for b in $(wildcard runs/doom/$(TRAIN_LEVEL)/seed$(DOOM_SEED)/*.json); do \
	  case $$b in *.meta.json) ;; *) $(UV) doomworm benchmark --brain $$b --maps viz$(LEVEL) $(DOOM_BENCH) --out-dir runs/benchmark_doom/train_$(TRAIN_LEVEL)/seed$(DOOM_SEED)/eval_viz$(LEVEL) || exit 1;; esac; done

# --- watching ----------------------------------------------------------------

watch-doom: ## Watch $(WATCH_BRAIN) play viz$(LEVEL) in the Doom window
	$(UV) doomworm play --brain $(WATCH_BRAIN) --maps viz$(LEVEL) --task doom --seed $(WATCH_SEED) --steps 600 --watch

watch-doomguy: ## Watch the hand-written floor, the one row that aims
	$(UV) doomworm benchmark --scripted doomguy --maps viz$(LEVEL) --task doom --sensors ideal --steps 600 --test-seeds 1 --repeats 1 --out-dir runs/watch --watch

watch-stock: ## Watch a brain play the stock scenario $(STOCK)
	$(UV) doomworm benchmark --brain $(WATCH_BRAIN) --maps $(STOCK) --task doom --sensors ideal --steps 600 --test-seeds 1 --repeats 1 --out-dir runs/watch --watch

# --- replay ------------------------------------------------------------------

play: ## Replay $(WORM_BRAIN) on seed $(SEED) with a plot
	$(UV) doomworm play --brain $(WORM_BRAIN) --seed $(SEED) --plot

stimulate: ## Stimulate $(NEURON) and plot propagation
	$(UV) doomworm stimulate $(NEURON) --plot
