#!/bin/bash
# Evolve lane: doom4 then doom6, seeds 0,1,2, candidates worm, worm_from_worm_evolved_random,
# worm_random, worm_shuffled, rnn (in that order). One evolve at a time. After each training
# run, immediately benchmark that brain against doom1..doom6.
set -uo pipefail
cd /Users/a.iuskasov/Workspace/DoomWorm
TRAINLOG=runs/doom/train.log

run_evolve () {
  local level=$1 seed=$2 name=$3 candidate=$4 extra=$5
  local outdir="runs/doom/${level}/seed${seed}"
  local out="${outdir}/${name}.json"
  local log="${outdir}/${name}.log"
  mkdir -p "$outdir"
  local cmd="uv run doomworm evolve --candidate ${candidate} --layer none --maps ${level} --task doom --sensors ideal --steps 600 --workers 6 --seed ${seed} ${extra} --out ${out}"
  local start end
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "=== START ${start} === ${cmd}" | tee -a "$TRAINLOG"
  eval "$cmd" > "$log" 2>&1
  local rc=$?
  end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  local bestline
  bestline=$(grep -E "best train fitness" "$log" | tail -1)
  echo "=== END ${end} rc=${rc} === ${cmd}" | tee -a "$TRAINLOG"
  echo "    ${bestline}" | tee -a "$TRAINLOG"
  if [ $rc -ne 0 ]; then
    echo "    CRASH: see ${log}" | tee -a "$TRAINLOG"
    tail -20 "$log" | tee -a "$TRAINLOG"
  fi
  echo "$rc"
}

benchmark_brain () {
  local level=$1 seed=$2 name=$3
  local brain="runs/doom/${level}/seed${seed}/${name}.json"
  if [ ! -f "$brain" ]; then
    echo "SKIP benchmark: ${brain} missing (training crashed)" | tee -a "$TRAINLOG"
    return
  fi
  for EVAL in doom1 doom2 doom3 doom4 doom5 doom6; do
    local outdir="runs/benchmark_doom/train_${level}/seed${seed}/eval_${EVAL}"
    mkdir -p "$outdir"
    uv run doomworm benchmark --brain "$brain" --maps "$EVAL" --task doom --sensors ideal --steps 600 --test-seeds 12 --repeats 1 --out-dir "$outdir" >> "runs/doom/${level}/seed${seed}/${name}.bench.log" 2>&1
  done
}

for LEVEL in doom4 doom6; do
  for SEED in 0 1 2; do
    run_evolve "$LEVEL" "$SEED" "worm" "worm" ""
    benchmark_brain "$LEVEL" "$SEED" "worm"

    run_evolve "$LEVEL" "$SEED" "worm_from_worm_evolved_random" "worm" "--init-brain docs/brains/a1/worm_evolved_random.json"
    benchmark_brain "$LEVEL" "$SEED" "worm_from_worm_evolved_random"

    run_evolve "$LEVEL" "$SEED" "worm_random" "worm_random" ""
    benchmark_brain "$LEVEL" "$SEED" "worm_random"

    run_evolve "$LEVEL" "$SEED" "worm_shuffled" "worm_shuffled" ""
    benchmark_brain "$LEVEL" "$SEED" "worm_shuffled"

    run_evolve "$LEVEL" "$SEED" "rnn" "rnn" ""
    benchmark_brain "$LEVEL" "$SEED" "rnn"

    run_evolve "$LEVEL" "$SEED" "ncp" "ncp" ""
    benchmark_brain "$LEVEL" "$SEED" "ncp"
  done
done
echo "=== EVOLVE LANE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$TRAINLOG"
