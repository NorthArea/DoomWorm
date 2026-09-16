"""Stage 22.1f: robustness sweep over the assumed sensor numbers."""

from dataclasses import replace
from pathlib import Path

from doomworm.environments.sensors import CAR, SensorConfig
from doomworm.layer import GradientFollower, PlannerLayer
from doomworm.learning import BenchmarkConfig, run_benchmark, sweep, sweep_table, worst_cells


def test_benchmark_accepts_an_explicit_sensor_config() -> None:
    cfg = BenchmarkConfig(sensors="car", test_seeds=(3000,), repeats=1, steps=40)
    blind = replace(CAR, dropout=1.0)
    layer = PlannerLayer(GradientFollower(), blind, mode="needs")
    result = run_benchmark(layer, "blind", replace(cfg, sensor_config=blind))
    assert result.rows[0].ticks == 40
    assert result.to_dict()["config"]["sensor_config"]["dropout"] == 1.0


def test_sweep_has_a_baseline_and_one_cell_per_changed_value(tmp_path: Path) -> None:
    def factory(cfg: SensorConfig) -> PlannerLayer:
        return PlannerLayer(GradientFollower(), cfg, mode="needs")

    bench = BenchmarkConfig(sensors="car", test_seeds=(3000,), repeats=1, steps=60)
    grid: dict[str, list[float | int]] = {"dropout": [0.05, 0.5], "delay": [1, 3]}  # first = preset
    rows = sweep(factory, "follower+needs", CAR, grid, bench)
    assert [(r.param, r.value) for r in rows] == [("base", 0), ("dropout", 0.5), ("delay", 3)]
    assert all(r.result.rows[0].ticks == 60 for r in rows)
    table = sweep_table(rows)
    assert "| base | (preset) |" in table
    assert "| dropout | 0.5 |" in table
    assert "| delay | 3 |" in table
    assert len(worst_cells(rows, 1)) == 1


def test_cli_robustness_writes_a_table(tmp_path: Path) -> None:
    from doomworm.cli import main

    out = tmp_path / "rb.md"
    args = ["robustness", "--scripted", "follower", "--test-seeds", "1", "--repeats", "1"]
    assert main([*args, "--steps", "30", "--params", "dropout", "--out", str(out)]) == 0
    text = out.read_text()
    assert "dropout" in text
    assert "largest drops" in text
    assert "beacon_fov" not in text, "only the requested parameter"
