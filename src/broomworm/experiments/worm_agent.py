"""Stage 9: the first worm-driven agent. The C. elegans connectome drives the 2D agent.

    2D world -> default sensory mapping -> 302 graded neurons -> group motor adapter -> wheels

No learning. Weights are the normalised dataset weights (stage 5); the only
hand-set inputs are tonic currents on AVBL/AVBR, the forward command
interneurons that are tonically active in the animal, and the gains of the
sensory channels. Everything is logged (Plan §15): position, heading, sensors,
active neurons, wheels, reward, collisions, food.

Run: ``uv run python -m broomworm.experiments.worm_agent [--seed N] [--plot] [--gif]``
"""

from __future__ import annotations

import argparse
import math
from itertools import pairwise
from pathlib import Path

import numpy as np

from broomworm.adapters import FireAdapter, GroupMotorAdapter, SensoryAdapter
from broomworm.brain import Network, Simulator
from broomworm.connectome import (
    Connectome,
    MotorMapping,
    SensoryMapping,
    build_network,
    default_motor_mapping,
    default_sensory_mapping,
    load_cook2019,
)
from broomworm.environments.maze import rooms_visited
from broomworm.environments.sensors import PRESETS, SensorSuite
from broomworm.environments.simple_2d import World
from broomworm.environments.worlds import build_world
from broomworm.episode import Record, run_episode
from broomworm.experiments.episode import print_trace, render_ascii, save_log, save_plot
from broomworm.learning import RewardTracker
from broomworm.visualization import save_debug_gif

SCENARIO_NAME = "worm"


class WormScenario:
    """Connectome brain + Plan §13/§14 mappings in the stage-4 world."""

    def __init__(
        self,
        *,
        connectome: Connectome | None = None,
        sensory_mapping: SensoryMapping | None = None,
        motor_mapping: MotorMapping | None = None,
        tonic_avb: float = 0.1,
        obstacle_gain: float = 3.0,
        food_gain: float = 1.0,
        gain: float = 0.45,
        decay: float = 0.5,
        gain_drive: float = 6.0,
        gain_turn: float = 6.0,
        gain_fire: float = 10.0,
        brain_steps: int = 5,
        maps: str = "fixed",
        task: str = "food",
        dangers: int = 0,
        sensors: str = "ideal",
    ) -> None:
        self.connectome = connectome or load_cook2019()
        self.sensory_mapping = sensory_mapping or default_sensory_mapping()
        self.motor_mapping = motor_mapping or default_motor_mapping()
        self.sensory_mapping.validate(self.connectome)
        self.motor_mapping.validate(self.connectome)
        self.sensory_mapping.scale("sensor_", obstacle_gain).scale("food_", food_gain)
        self.sensory_mapping.tonic.update({"AVBL": tonic_avb, "AVBR": tonic_avb})
        self.template: Network = build_network(self.connectome, gain=gain, decay=decay)
        self.sensory: SensoryAdapter = self.sensory_mapping.to_adapter()
        m = self.motor_mapping
        self.motor = GroupMotorAdapter(
            m.forward, m.reversal, m.turn_left, m.turn_right, gain_drive, gain_turn
        )
        # the trigger (Plan §22) reads the pharyngeal group; absent when the mapping has none
        self.trigger: FireAdapter | None = FireAdapter(m.fire, gain_fire) if m.fire else None
        self.brain_steps = brain_steps
        if maps not in ("fixed", "random", "apartment") and not maps.startswith("room:"):
            raise ValueError("maps must be 'fixed', 'random', 'apartment' or 'room:<file>'")
        self.maps = maps
        if task not in ("food", "target", "clean"):
            raise ValueError("task must be 'food', 'target' or 'clean'")
        self.task = task
        self.dangers = dangers
        if sensors not in PRESETS:
            raise ValueError(f"sensors must be one of {list(PRESETS)}")
        self.sensors = sensors
        self.params = {
            "tonic_avb": tonic_avb,
            "obstacle_gain": obstacle_gain,
            "food_gain": food_gain,
            "gain": gain,
            "decay": decay,
            "gain_drive": gain_drive,
            "gain_turn": gain_turn,
            "gain_fire": gain_fire,
            "brain_steps": brain_steps,
            "maps": maps,
            "task": task,
            "dangers": dangers,
            "sensors": sensors,
        }

    @property
    def n_weights(self) -> int:
        """Genome length (stage 10)."""
        return len(self.template.synapses)

    def random_weights(self, seed: int) -> np.ndarray:
        """A random genome with the same scale as the initial weights."""
        base = np.array(self.template.get_weights())
        rng = np.random.default_rng(seed)
        return np.asarray(rng.uniform(-1.0, 1.0, size=base.size) * np.abs(base).max())

    def make_sensors(self, seed: int) -> SensorSuite | None:
        """A fresh sensor suite for an episode (None for the ideal preset)."""
        if self.sensors == "ideal":
            return None
        return SensorSuite(PRESETS[self.sensors], seed=seed)

    def make_world(self, seed: int) -> World:
        """See :func:`build_world`."""
        return build_world(seed, self.maps, self.task, self.dangers)


def run_worm(
    scenario: WormScenario, seed: int, steps: int, record_activity: bool = True
) -> tuple[World, list[Record], RewardTracker]:
    """One logged episode of the untrained worm."""
    world = scenario.make_world(seed)
    scenario.template.reset()
    tracker = RewardTracker()
    trace = run_episode(
        world,
        Simulator(scenario.template),
        scenario.sensory,
        scenario.motor,
        steps,
        tracker,
        record_activity=record_activity,
        brain_steps=scenario.brain_steps,
        sensors=scenario.make_sensors(seed),
    )
    return world, trace, tracker


def summarise(trace: list[Record], world: World) -> dict[str, float]:
    """Numbers worth comparing across seeds and, later, training runs."""
    distance = sum(math.dist((a.x, a.y), (b.x, b.y)) for a, b in pairwise(trace))
    return {
        "ticks": len(trace),
        "distance": distance,
        "food": world.food_eaten,
        "targets": world.targets_reached,
        "damage": world.damage_taken,
        "rooms": rooms_visited(world, [(r.x, r.y) for r in trace]),
        "coverage": world.coverage,
        "dockings": world.dockings,
        "charging": world.charging_ticks,
        "collisions": world.collisions,
        "kills": world.kills,
        "hits": world.hits,
        "shots": world.shots,
        "exited": int(world.exited),
        "reward": sum(r.reward for r in trace),
        "mean_left": float(np.mean([r.motors[0] for r in trace])),
        "mean_right": float(np.mean([r.motors[1] for r in trace])),
        "reversing": sum(1 for r in trace if r.motors[0] < 0 and r.motors[1] < 0),
    }


def main(argv: list[str] | None = None) -> int:
    """Run one episode, print the trace, map and summary; write logs to runs/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--every", type=int, default=25)
    parser.add_argument("--tonic-avb", type=float, default=0.1)
    parser.add_argument("--obstacle-gain", type=float, default=3.0)
    parser.add_argument("--maps", choices=["fixed", "random", "apartment"], default="fixed")
    parser.add_argument("--task", choices=["food", "target", "clean"], default="food")
    parser.add_argument("--dangers", type=int, default=0)
    parser.add_argument("--sensors", choices=list(PRESETS), default="ideal")
    parser.add_argument("--plot", action="store_true", help="save runs/stage9_worm_<seed>.png")
    parser.add_argument("--gif", action="store_true", help="save runs/stage9_worm_<seed>.gif")
    args = parser.parse_args(argv)

    scenario = WormScenario(
        tonic_avb=args.tonic_avb,
        obstacle_gain=args.obstacle_gain,
        maps=args.maps,
        task=args.task,
        dangers=args.dangers,
        sensors=args.sensors,
    )
    world, trace, tracker = run_worm(scenario, args.seed, args.steps)
    print_trace(trace, args.every)
    print()
    print(render_ascii(world, trace))
    summary = summarise(trace, world)
    cells = [f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in summary.items()]
    print("\n" + "  ".join(cells))
    print(f"reward breakdown {tracker.breakdown}")

    runs = Path("runs")
    log = runs / f"stage9_worm_{args.seed}.jsonl"
    save_log(trace, log)
    print(f"saved {log}")
    if args.plot:
        out = runs / f"stage9_worm_{args.seed}.png"
        save_plot(world, trace, out, f"untrained worm, seed {args.seed}")
        print(f"saved {out}")
    if args.gif:
        out = runs / f"stage9_worm_{args.seed}.gif"
        frames = save_debug_gif(world, trace, out, every=5)
        print(f"saved {out} ({frames} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
