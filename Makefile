# DoomWorm developer shortcuts. Everything runs through uv; nothing is installed globally.

.DEFAULT_GOAL := help
UV := uv run
BRAIN ?= runs/small_evolved.json
SEED ?= 1003
NEURON ?= ASHL

.PHONY: help sync hooks test test-fast cov lint format typecheck check clean \
	    demo-0 demo-1 demo-2 demo-3 demo-5 demo-6 demo-7 demo-8 demo-9 demo-12 demo-13 demo-14 demo-15 demo-16 demo-17 demo-18 demo-19 demo-20 demo-21 demos train train-worm train-worm-random train-worm-danger train-worm-apartment benchmark benchmark-a2 compare evolve-a2 play stimulate fetch-data

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

demo-9: ## Stage 9: untrained worm drives the 2D agent (PNG + GIF + JSONL log)
	$(UV) python -m doomworm.experiments.worm_agent --seed $(SEED) --plot --gif

demo-12: ## Stage 12: brain trained on random maps, played on an unseen random map
	$(UV) doomworm play --brain runs/worm_evolved_random.json --seed $(SEED) --plot

demo-13: ## Stage 13: the same brain sent to a target ("come to X") instead of food
	$(UV) doomworm play --brain runs/worm_evolved_random.json --task target --seed $(SEED) --plot

demo-16: ## Stage 16: vacuum task (dirt, dock, battery) on an unseen apartment, stage-12 brain
	$(UV) doomworm play --brain runs/worm_evolved_random.json --maps apartment --task clean --seed $(SEED) --steps 800 --plot

demo-15: ## Stage 15: a brain on an unseen apartment map (rooms, doors, furniture)
	$(UV) doomworm play --brain runs/worm_evolved_random.json --maps apartment --seed $(SEED) --plot

demo-14: ## Stage 14: target task with a danger zone (runs/worm_evolved_danger.json)
	$(UV) doomworm play --brain runs/worm_evolved_danger.json --seed $(SEED) --plot

WORM_BRAIN ?= docs/results/brains/worm_evolved_random.json

demo-17: ## Stage 17: stage-12 brain under the noisy vacuum sensor suite
	$(UV) doomworm play --brain $(WORM_BRAIN) --maps random --sensors noisy --seed $(SEED) --plot

demo-18: ## Stage 18: benchmark the stage-12 brain alone -> runs/benchmark/
	$(UV) doomworm benchmark --brain $(WORM_BRAIN) --name worm_stage12

demo-19: ## Stage 19: the same brain wrapped in the coverage planner
	$(UV) doomworm benchmark --brain $(WORM_BRAIN) --name worm_stage12 --planner coverage

demo-20: ## Stage 20: needs arbitration (battery > call > clean), worm and scripted driver
	$(UV) doomworm benchmark --brain $(WORM_BRAIN) --name worm_stage12 --planner needs
	$(UV) doomworm benchmark --scripted follower --planner needs

demo-21: ## Stage 21.1: the Roomba-style classical controller, zero learning, on the benchmark
	$(UV) doomworm benchmark --scripted roomba

demos: demo-0 demo-1 demo-2 demo-3 demo-5 demo-6 demo-7 demo-8 demo-9 ## Run every stage demo

# --- training and connectome -------------------------------------------------

train: ## Stage 4: evolve the small network -> $(BRAIN)
	$(UV) doomworm train --out $(BRAIN)

train-worm: ## Stage 10: evolve the connectome weights -> runs/worm_evolved.json
	$(UV) doomworm train --scenario worm --out runs/worm_evolved.json $(TRAIN_ARGS)

train-worm-random: ## Stage 12: evolve the connectome on random maps -> runs/worm_evolved_random.json
	$(UV) doomworm train --scenario worm --maps random --out runs/worm_evolved_random.json $(TRAIN_ARGS)

train-worm-danger: ## Stage 14: continue from the random-map brain on target + danger
	$(UV) doomworm train --scenario worm --maps random --task target --dangers 1 --init-brain runs/worm_evolved_random.json --out runs/worm_evolved_danger.json $(TRAIN_ARGS)

train-worm-apartment: ## Stage 15: continue from the random-map brain on apartment maps
	$(UV) doomworm train --scenario worm --maps apartment --init-brain runs/worm_evolved_random.json --out runs/worm_evolved_apartment.json $(TRAIN_ARGS)

benchmark: ## Stage 18: benchmark $(BRAIN) on apartments/clean/vacuum -> runs/benchmark/
	$(UV) doomworm benchmark --brain $(BRAIN) $(BENCH_ARGS)

CANDIDATE ?= worm
WORKERS ?= 4

evolve-a2: ## Stage 21.2: train $(CANDIDATE) (worm|worm_random|worm_shuffled|worm_dense) under the needs layer -> runs/a2/
	$(UV) doomworm evolve --candidate $(CANDIDATE) --workers $(WORKERS) $(EVOLVE_ARGS)

benchmark-a2: ## Stage 21: every trained candidate in runs/a2/ plus the scripted floors -> runs/benchmark_a2/
	$(UV) doomworm benchmark --scripted roomba --out-dir runs/benchmark_a2
	$(UV) doomworm benchmark --scripted follower --planner needs --out-dir runs/benchmark_a2
	for b in runs/a2/*.json; do case $$b in *.meta.json) ;; *) \
	  $(UV) doomworm benchmark --brain $$b --planner needs --out-dir runs/benchmark_a2 || exit 1 ;; esac; done

compare: ## Stage 11: real vs random vs shuffled vs free -> runs/compare/
	$(UV) doomworm compare $(COMPARE_ARGS)

play: ## Replay $(BRAIN) on seed $(SEED) with a plot
	$(UV) doomworm play --brain $(BRAIN) --seed $(SEED) --plot

stimulate: ## Stage 6: stimulate $(NEURON) and plot propagation
	$(UV) doomworm stimulate $(NEURON) --plot

fetch-data: ## Re-download the Cook 2019 connectome into data/connectome/cook2019
	uv run --with openpyxl scripts/fetch_connectome.py
