#!/bin/bash
# PPO lane: doom4 seeds 0-2 then doom6 seeds 0-2. One ppo at a time. After each training run,
# immediately benchmark that brain against doom1..doom6.
set -uo pipefail
cd /Users/a.iuskasov/Workspace/DoomWorm
TRAINLOG=runs/doom/train.log

run_ppo () {
  local level=$1 seed=$2
  local outdir="runs/doom/${level}/seed${seed}"
  local out="${outdir}/ppo.json"
  local log="${outdir}/ppo.log"
  mkdir -p "$outdir"
  local cmd="uv run doomworm ppo --layer none --maps ${level} --task doom --sensors ideal --steps 600 --seed ${seed} --out ${out}"
  local start end
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "=== START ${start} === ${cmd}" | tee -a "$TRAINLOG"
  eval "$cmd" > "$log" 2>&1
  local rc=$?
  end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  local bestline
  bestline=$(grep -E "ep_rew_mean" "$log" | tail -1)
  echo "=== END ${end} rc=${rc} === ${cmd}" | tee -a "$TRAINLOG"
  echo "    last ep_rew_mean: ${bestline}" | tee -a "$TRAINLOG"
  if [ $rc -ne 0 ]; then
    echo "    CRASH: see ${log}" | tee -a "$TRAINLOG"
    tail -20 "$log" | tee -a "$TRAINLOG"
  fi
}

benchmark_brain () {
  local level=$1 seed=$2
  local brain="runs/doom/${level}/seed${seed}/ppo.json"
  if [ ! -f "$brain" ]; then
    echo "SKIP benchmark: ${brain} missing (training crashed)" | tee -a "$TRAINLOG"
    return
  fi
  for EVAL in doom1 doom2 doom3 doom4 doom5 doom6; do
    local outdir="runs/benchmark_doom/train_${level}/seed${seed}/eval_${EVAL}"
    mkdir -p "$outdir"
    uv run doomworm benchmark --brain "$brain" --maps "$EVAL" --task doom --sensors ideal --steps 600 --test-seeds 12 --repeats 1 --out-dir "$outdir" >> "runs/doom/${level}/seed${seed}/ppo.bench.log" 2>&1
  done
}

for LEVEL in doom4 doom6; do
  for SEED in 0 1 2; do
    run_ppo "$LEVEL" "$SEED"
    benchmark_brain "$LEVEL" "$SEED"
  done
done
echo "=== PPO LANE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$TRAINLOG"
