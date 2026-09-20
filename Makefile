# WormLab developer shortcuts. Everything runs through uv; nothing is installed globally.

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
WORM_BRAIN ?= docs/brains/curriculum/worm_evolved_random.json
A2_WORM ?= docs/broom/brains/a2/worm_from_worm_evolved_random.json
BROOM_BRAIN ?= $(A2_WORM)
LINK ?= fake
WATCH_BRAIN ?= docs/doom/brains/doom/doom4/seed0/worm_from_worm_evolved_random.json
WATCH_SEED ?= 3000
DOOM_BENCH = --task doom --sensors ideal --steps 600 --test-seeds 12 --repeats 1

.PHONY: help tracks broom-arduino-setup broom-firmware broom-firmware-flash broom-motor-map \
	    broom-selftest broom-calibrate broom-robustness broom-benchmark broom-evolve-a2 broom-benchmark-a2 \
	    broom-evolve-car broom-benchmark-car broom-drive broom-dayone setup sync hooks fetch-data fetch-doom test test-fast cov lint format typecheck check clean \
	    demo-0 demo-1 demo-2 demo-3 demo-5 demo-6 demo-7 demo-8 demo-9 demos \
	    demo-b1 demo-b4 demo-b11 demo-classic watch-doom watch-doomguy watch-stock watch-classic \
	    benchmark-doom benchmark-vizdoom benchmark-classic evolve-doom train-worm compare play stimulate \
	    lane-bakeoff lane-memory report

tracks: ## What lives where: two tracks, one platform
	@echo "  src/wormlab/    the platform both tracks share  (changes rarely)"
	@echo "  src/doomworm/   the Doom player                 (doom-* targets, docs/doom/)"
	@echo "  src/broomworm/  the home robot                  (broom-* targets, docs/broom/)"
	@echo "  docs/knowledge/ what either track learned that the other can use"

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- setup -------------------------------------------------------------------

setup: sync fetch-data fetch-doom ## Everything a fresh clone needs: venv, connectome, Doom data
	@echo "ready: try 'make demo-b1' or 'make watch-doomguy'"

sync: ## Install/refresh the venv from uv.lock (all groups)
	uv sync --all-groups

hooks: sync ## Install pre-commit hooks into .git
	$(UV) pre-commit install

fetch-data: ## Download the connectome dataset
	$(UV) python scripts/fetch_connectome.py

fetch-doom: ## Download the Freedoom IWADs, needed for the classic maps e1m1..e4m9
	$(UV) scripts/fetch_freedoom.py

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
	$(UV) python -m wormlab.experiments.three_neurons

demo-1: ## Obstacle avoidance (PNG in runs/)
	$(UV) python -m wormlab.experiments.obstacle_agent --plot

demo-2: ## The food task: the curriculum world (PNG in runs/)
	$(UV) python -m wormlab.experiments.food_agent --plot

demo-3: ## Reward comparison of three brains
	$(UV) python -m wormlab.experiments.reward_demo

demo-5: ## Connectome statistics
	$(UV) python -m wormlab.experiments.connectome_stats

demo-6: ## Debug screen GIF (runs/stage6_debug.gif)
	$(UV) python -m wormlab.experiments.debug_screen_demo

demo-7: ## Sensory mapping and per-channel responders
	$(UV) python -m wormlab.experiments.sensory_mapping_demo

demo-8: ## Motor mapping, commands per channel
	$(UV) python -m wormlab.experiments.motor_mapping_demo

demo-9: ## The untrained worm drives the agent (PNG + GIF + JSONL log)
	$(UV) python -m wormlab.experiments.worm_agent --seed $(SEED) --plot --gif

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

CLASSIC ?= e1m1

demo-classic: ## The classic game: the floor and a trained worm on $(CLASSIC) (needs make fetch-doom)
	$(UV) doomworm benchmark --scripted doomguy --maps $(CLASSIC) --task doom --sensors ideal --steps 600 --test-seeds 3 --repeats 1 --out-dir runs/classic
	$(UV) doomworm benchmark --brain $(WATCH_BRAIN) --maps $(CLASSIC) --task doom --sensors ideal --steps 600 --test-seeds 3 --repeats 1 --out-dir runs/classic

benchmark-classic: ## Every brain of runs/doom/$(TRAIN_LEVEL)/seed$(DOOM_SEED)/ on the classic map $(CLASSIC)
	$(UV) doomworm benchmark --scripted doomguy --maps $(CLASSIC) $(DOOM_BENCH) --out-dir runs/classic
	for b in $(wildcard runs/doom/$(TRAIN_LEVEL)/seed$(DOOM_SEED)/*.json); do \
	  case $$b in *.meta.json) ;; *) $(UV) doomworm benchmark --brain $$b --maps $(CLASSIC) $(DOOM_BENCH) --out-dir runs/classic || exit 1;; esac; done

# --- experiment lanes (hours; they resume if interrupted) ---------------------

lane-bakeoff: ## Train every candidate on doom4 and doom6, three seeds, then benchmark them all
	$(UV) scripts/doom_b2_resume.py

lane-memory: ## Stage 16: the worm and its shuffle, with and without the memory layer
	$(UV) scripts/memory_lane.py

report: ## Rebuild the result tables in docs/doom/results/b/ from the benchmark rows
	$(UV) python scripts/doom_report.py runs/benchmark_doom --out docs/doom/results/b/b2_doom.md


# --- the robot track ----------------------------------------------------------

broom-arduino-setup: ## Stage 23.2: project-local arduino-cli + ESP32 core + ESP32Servo (no global install)
	mkdir -p .tools/arduino && test -x .tools/arduino-cli || (curl -sSL https://downloads.arduino.cc/arduino-cli/arduino-cli_latest_macOS_ARM64.tar.gz | tar xz -C .tools arduino-cli)
	$(ARDUINO) core update-index && $(ARDUINO) core install esp32:esp32 && $(ARDUINO) lib install ESP32Servo

broom-firmware: ## Stage 23.2: compile firmware/esp32_car for the ESP32 Max V1.0 (ESP32 Dev Module)
	$(ARDUINO) compile --fqbn $(FQBN) --warnings default firmware/esp32_car

broom-firmware-flash: ## Stage 23.2: flash the car (GPIO0 "00" to GND + RST first; PORT=/dev/cu.usbserial-XXXX)
	$(ARDUINO) upload --fqbn $(FQBN) --port $(PORT) firmware/esp32_car

broom-motor-map: ## Stage 23.2 bench: map the shield's shift-register bits to wheels (LINK=fake|tcp)
	$(UV) broomworm motor-map --link $(LINK)

broom-selftest: ## Day one on the machine: protocol, sensors, wheels (LINK=sim|tcp)
	$(UV) broomworm selftest --link $(LINK)

broom-calibrate: ## Measure metres per world unit and the wheel base (LINK=sim|tcp)
	$(UV) broomworm calibrate --link $(LINK)

broom-robustness: ## Sweep the sensor preset's assumed numbers around the scripted robot
	$(UV) broomworm robustness --scripted roomba

broom-benchmark: ## One brain through this track's benchmark (BROOM_BRAIN=..., ARGS=...)
	$(UV) broomworm benchmark --brain $(BROOM_BRAIN) --planner needs $(ARGS)

broom-evolve-a2: ## Train $(CANDIDATE) under the needs layer on the vacuum preset -> runs/a2/
	$(UV) broomworm evolve --candidate $(CANDIDATE) --workers $(WORKERS) $(EVOLVE_ARGS)

broom-benchmark-a2: ## The A2 table: every brain in runs/a2/ plus the scripted floors -> runs/benchmark_a2/
	$(UV) broomworm benchmark --scripted roomba --out-dir runs/benchmark_a2
	$(UV) broomworm benchmark --scripted follower --planner needs --out-dir runs/benchmark_a2
	for b in runs/a2/*.json; do case $$b in *.meta.json) ;; *) \
	  $(UV) broomworm benchmark --brain $$b --planner needs --out-dir runs/benchmark_a2 || exit 1 ;; esac; done

broom-evolve-car: ## Retrain $(CANDIDATE) on the car preset -> runs/a2_car/ (PPO: broomworm ppo --sensors car)
	$(UV) broomworm evolve --candidate $(CANDIDATE) --sensors car --workers $(WORKERS) --out runs/a2_car/$(CANDIDATE).json

broom-benchmark-car: ## The A3 table: the A2 brains and the floors on the car preset -> runs/benchmark_car/
	$(UV) broomworm benchmark --scripted roomba --sensors car --out-dir runs/benchmark_car
	$(UV) broomworm benchmark --scripted follower --planner needs --sensors car --out-dir runs/benchmark_car
	for b in worm worm_from_worm_evolved_random rnn ppo; do \
	  $(UV) broomworm benchmark --brain docs/broom/brains/a2/$$b.json --planner needs --sensors car \
	    --out-dir runs/benchmark_car || exit 1; done
	for b in runs/a2_car/*.json; do case $$b in *.meta.json) ;; runs/a2_car/\*.json) ;; *) \
	  $(UV) broomworm benchmark --brain $$b --name $$(basename $$b .json)_car --planner needs \
	    --sensors car --out-dir runs/benchmark_car || exit 1 ;; esac; done

broom-drive: ## A brain over the simulator link, recorded, then replayed and compared (stage 22.1)
	$(UV) broomworm drive --brain $(A2_WORM) --planner needs --seed 3002 --steps 300 --every 50 \
	  --record runs/drive/worm_sim_3002.jsonl
	$(UV) broomworm compare-log --log runs/drive/worm_sim_3002.jsonl --out runs/drive/worm_sim_3002_vs_replay.md
	$(UV) broomworm plot-log --log runs/drive/worm_sim_3002.jsonl

broom-dayone: ## Rehearse the whole day on the car, on the simulator: bench, self-test, calibrate, room drive, compare
	$(UV) broomworm motor-map --link fake --ms 100
	$(UV) broomworm selftest --link sim --sensors car --seed 3001 --out runs/dayone/selftest.md
	printf '0.8\n60\n' | $(UV) broomworm calibrate --link sim --sensors car --seed 3001 --out runs/dayone/calibration.json
	printf 'w\nw\nw\nw\nd\nd\nw\nw\nw\n' | $(UV) broomworm drive --link sim --teleop --sensors car \
	  --room data/rooms/example_room.json --every 1 --record runs/dayone/room.jsonl
	$(UV) broomworm plot-log --log runs/dayone/room.jsonl
	$(UV) broomworm compare-log --log runs/dayone/room.jsonl --out runs/dayone/room_vs_replay.md
	@echo "rehearsed: on the car the same lines run with --link tcp (docs/broom/hardware.md)"

# --- watching ----------------------------------------------------------------

watch-doom: ## Watch $(WATCH_BRAIN) play viz$(LEVEL) in the Doom window
	$(UV) doomworm play --brain $(WATCH_BRAIN) --maps viz$(LEVEL) --task doom --seed $(WATCH_SEED) --steps 600 --watch

watch-doomguy: ## Watch the hand-written floor, the one row that aims
	$(UV) doomworm benchmark --scripted doomguy --maps viz$(LEVEL) --task doom --sensors ideal --steps 600 --test-seeds 1 --repeats 1 --out-dir runs/watch --watch

watch-stock: ## Watch a brain play the stock scenario $(STOCK)
	$(UV) doomworm benchmark --brain $(WATCH_BRAIN) --maps $(STOCK) --task doom --sensors ideal --steps 600 --test-seeds 1 --repeats 1 --out-dir runs/watch --watch

watch-classic: ## Watch a brain play the classic map $(CLASSIC) (needs make fetch-doom)
	$(UV) doomworm benchmark --brain $(WATCH_BRAIN) --maps $(CLASSIC) --task doom --sensors ideal --steps 600 --test-seeds 1 --repeats 1 --out-dir runs/watch --watch

watch-classic-floor: ## Watch the hand-written floor play the classic map $(CLASSIC)
	$(UV) doomworm benchmark --scripted doomguy --maps $(CLASSIC) --task doom --sensors ideal --steps 600 --test-seeds 1 --repeats 1 --out-dir runs/watch --watch

# --- replay ------------------------------------------------------------------

play: ## Replay $(WORM_BRAIN) on seed $(SEED) with a plot
	$(UV) doomworm play --brain $(WORM_BRAIN) --seed $(SEED) --plot

stimulate: ## Stimulate $(NEURON) and plot propagation
	$(UV) doomworm stimulate $(NEURON) --plot
