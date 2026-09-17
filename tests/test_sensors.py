"""Stage 17: vacuum sensor suite with noise, dropout, delay, odometry (Plan §20.3)."""

import math

import pytest

from doomworm.environments.sensors import IDEAL, PRESETS, SensorConfig, SensorSuite
from doomworm.environments.simple_2d import AgentState, Danger, Obstacle, Wall, World
from doomworm.episode import run_episode
from doomworm.experiments import food_agent
from doomworm.experiments.worm_agent import WormScenario


def world_with_wall_ahead() -> World:
    return World(agent=AgentState(x=5.0, y=10.0), walls=[Wall(8.0, 0.0, 1.0, 20.0)])


def test_ideal_suite_reproduces_legacy_channels() -> None:
    w = World(agent=AgentState(x=10.0, y=10.0), obstacles=[Obstacle(x=13.0, y=10.0, radius=1.0)])
    suite = SensorSuite(IDEAL)
    suite.reset(w)
    obs = w.observe()
    ch = suite.read(w, obs)
    assert ch["sensor_front"] == pytest.approx(obs.sensor_front)
    assert ch["sensor_left"] == pytest.approx(obs.sensor_left)
    assert ch["range_1"] == pytest.approx(obs.sensor_front)
    assert set(suite.channel_names) == set(ch)
    assert ch["bumper_left"] == 0.0, "ideal = noiseless, every vacuum sensor present"
    assert ch["odom_x"] == pytest.approx(w.agent.x)


def test_dropout_zeroes_readings() -> None:
    cfg = SensorConfig(name="drop", dropout=1.0)
    w = world_with_wall_ahead()
    suite = SensorSuite(cfg)
    suite.reset(w)
    ch = suite.read(w, w.observe())
    assert ch["sensor_front"] == 0.0
    assert ch["range_1"] == 0.0


def test_delay_shifts_readings() -> None:
    cfg = SensorConfig(name="lag", delay=2)
    w = World(agent=AgentState(x=2.0, y=10.0), walls=[Wall(6.0, 0.0, 1.0, 20.0)], speed=1.0)
    suite = SensorSuite(cfg)
    suite.reset(w)
    reads = []
    for _ in range(4):
        obs = w.step(1.0, 1.0)
        reads.append(suite.read(w, obs)["sensor_front"])
    assert reads[:2] == [0.0, 0.0]
    assert reads[2] > 0.0
    assert reads[3] > reads[2], "readings lag the true approach by two steps"


def test_bumper_reports_contact_side() -> None:
    cfg = SensorConfig(name="bump", bumper=True, ray_angles=IDEAL.ray_angles)
    w = World(agent=AgentState(x=7.4, y=10.0), walls=[Wall(8.0, 0.0, 1.0, 20.0)], speed=0.5)
    suite = SensorSuite(cfg)
    suite.reset(w)
    obs = w.step(1.0, 1.0)
    ch = suite.read(w, obs)
    assert obs.collided
    assert (ch["bumper_left"], ch["bumper_right"]) == (1.0, 1.0), "head-on: both"
    w2 = World(
        agent=AgentState(x=7.4, y=10.0, heading=0.3), walls=[Wall(8.0, 0.0, 1.0, 20.0)], speed=0.5
    )
    s2 = SensorSuite(cfg)
    s2.reset(w2)
    ch2 = s2.read(w2, w2.step(1.0, 1.0))
    assert (ch2["bumper_left"], ch2["bumper_right"]) == (0.0, 1.0), "wall on the right side"


def test_cliff_sensor_sees_hazard_ahead() -> None:
    cfg = SensorConfig(name="cliff", cliff=True)
    w = World(agent=AgentState(x=5.0, y=10.0), dangers=[Danger(x=5.7, y=10.4, radius=0.4)])
    suite = SensorSuite(cfg)
    suite.reset(w)
    ch = suite.read(w, w.observe())
    assert ch["cliff_left"] == 1.0
    assert ch["cliff_right"] == 0.0


def test_wall_sensor_on_the_right() -> None:
    cfg = SensorConfig(name="wall", wall_sensor=True, wall_range=1.0)
    w = World(agent=AgentState(x=5.0, y=10.6), walls=[Wall(0.0, 9.0, 20.0, 1.0)])  # wall below
    suite = SensorSuite(cfg)
    suite.reset(w)
    assert suite.read(w, w.observe())["wall_right"] == pytest.approx(0.4)


def test_odometry_tracks_and_drifts() -> None:
    exact = SensorSuite(SensorConfig(name="odo", odometry=True))
    noisy = SensorSuite(
        SensorConfig(name="odo2", odometry=True, odom_sigma=0.2, gyro_drift=0.05), seed=3
    )
    for suite in (exact, noisy):
        w = World(agent=AgentState(x=5.0, y=5.0), speed=0.5)
        suite.reset(w)
        for _ in range(20):
            obs = w.step(1.0, 0.6)
            ch = suite.read(w, obs)
        err = math.dist((ch["odom_x"], ch["odom_y"]), (w.agent.x, w.agent.y))
        if suite is exact:
            assert err < 1e-6
            assert ch["odom_heading"] == pytest.approx(w.agent.heading)
            assert ch["gyro_heading"] == pytest.approx(w.agent.heading)
        else:
            assert err > 0.05
            assert ch["gyro_heading"] != pytest.approx(w.agent.heading)


def test_episode_runs_with_a_suite_and_scenario_param() -> None:
    world, sim, sensory, motor = food_agent.build_scenario()
    noisy = SensorConfig(name="noisy", noise_sigma=0.1, dropout=0.05)
    trace = run_episode(world, sim, sensory, motor, 30, sensors=SensorSuite(noisy, seed=0))
    assert len(trace) == 30
    sc = WormScenario(sensors="ideal")
    assert sc.params["sensors"] == "ideal"
    assert sc.make_sensors(1) is None, "the ideal preset is noiseless: the loop reads the world"
    with pytest.raises(ValueError, match="sensors"):
        WormScenario(sensors="lidar")
    assert set(PRESETS) == {"ideal"}
