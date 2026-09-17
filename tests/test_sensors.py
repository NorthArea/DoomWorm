"""Stage 17: vacuum sensor suite with noise, dropout, delay, odometry (Plan §20.3)."""

import math

import pytest

from broomworm.environments.sensors import IDEAL, NOISY, PRESETS, VACUUM, SensorConfig, SensorSuite
from broomworm.environments.simple_2d import AgentState, Danger, Obstacle, Wall, World
from broomworm.episode import run_episode
from broomworm.experiments import food_agent
from broomworm.experiments.worm_agent import WormScenario


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


def test_vacuum_channels_and_noise_are_seeded() -> None:
    w = world_with_wall_ahead()
    a = SensorSuite(VACUUM, seed=1)
    b = SensorSuite(VACUUM, seed=1)
    c = SensorSuite(VACUUM, seed=2)
    for s in (a, b, c):
        s.reset(w)
    obs = w.observe()
    # delay 1: the first read is the warm reading taken at rest (stale, not zero)
    first = a.read(w, obs)
    assert abs(first["range_2"] - obs.sensor_front) < 0.3
    assert first["hunger"] == obs.hunger
    b.read(w, obs)
    c.read(w, obs)
    ra, rb, rc = a.read(w, obs), b.read(w, obs), c.read(w, obs)
    assert ra == rb
    assert ra["range_2"] != rc["range_2"], "different seeds, different noise"
    assert abs(ra["range_2"] - obs.sensor_front) < 0.3
    assert {"bumper_left", "cliff_left", "wall_right", "odom_x", "gyro_heading"} <= set(ra)


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
    cfg = SensorConfig(name="bump", bumper=True, ray_angles=VACUUM.ray_angles)
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
    trace = run_episode(world, sim, sensory, motor, 30, sensors=SensorSuite(NOISY, seed=0))
    assert len(trace) == 30
    sc = WormScenario(sensors="vacuum")
    assert sc.params["sensors"] == "vacuum"
    assert isinstance(sc.make_sensors(1), SensorSuite)
    assert WormScenario().make_sensors(1) is None
    with pytest.raises(ValueError, match="sensors"):
        WormScenario(sensors="lidar")
    assert set(PRESETS) == {"ideal", "vacuum", "noisy", "car"}


# --- stage 22: the kit car preset ------------------------------------------------------


def test_car_preset_has_the_vacuum_channel_layout_with_three_rays() -> None:
    from broomworm.environments.sensors import CAR

    names = SensorSuite(CAR).channel_names
    assert [n for n in names if n.startswith("range_")] == ["range_0", "range_1", "range_2"]
    for key in ("bumper_left", "cliff_right", "wall_right", "odom_x", "gyro_heading", "dock_front"):
        assert key in names


def test_car_sweep_refreshes_one_ray_per_tick() -> None:
    from broomworm.environments.sensors import CAR

    cfg = SensorConfig(name="sweep", ray_angles=CAR.ray_angles, sweep=True)
    w = World(agent=AgentState(x=5.0, y=10.0, heading=0.0), walls=[Wall(8.0, 0.0, 1.0, 20.0)])
    suite = SensorSuite(cfg)
    suite.reset(w)
    first = suite.read(w, w.observe())
    # spin 90 degrees in one go: only the ray measured this tick sees the new geometry
    w.agent = AgentState(x=5.0, y=10.0, heading=math.pi / 2)
    second = suite.read(w, w.observe())
    changed = [i for i in range(3) if second[f"range_{i}"] != first[f"range_{i}"]]
    assert len(changed) == 1, "one servo position per tick"
    for _ in range(2):
        second = suite.read(w, w.observe())
    exact = [
        max(0.0, 1.0 - w.ray_distance(w.agent.heading + a) / cfg.ray_range) for a in cfg.ray_angles
    ]
    assert [second[f"range_{i}"] for i in range(3)] == pytest.approx(exact), "a full sweep later"


def test_car_proximity_bumper_fires_before_contact_and_ignores_real_collisions() -> None:
    from dataclasses import replace

    from broomworm.environments.sensors import CAR

    cfg = SensorConfig(name="pb", ray_angles=CAR.ray_angles, proximity_bumper=0.7)
    w = World(agent=AgentState(x=7.0, y=10.0, heading=0.0), walls=[Wall(8.0, 0.0, 1.0, 20.0)])
    suite = SensorSuite(cfg)
    suite.reset(w)
    far = suite.read(w, w.observe())
    assert (far["bumper_left"], far["bumper_right"]) == (0.0, 0.0), "1.0 away: no hit"
    w.agent = AgentState(x=7.4, y=10.0, heading=0.0)
    near = suite.read(w, w.observe())
    assert (near["bumper_left"], near["bumper_right"]) == (1.0, 1.0), "front ray under 0.7"
    # a real collision from the side is invisible to a ray-based bumper
    w.agent = AgentState(x=5.0, y=10.0, heading=math.pi / 2)
    side = suite.read(w, replace(w.observe(), collided=True))
    assert (side["bumper_left"], side["bumper_right"]) == (0.0, 0.0)


def test_car_command_odometry_keeps_moving_against_a_wall_and_gyro_follows_it() -> None:
    from broomworm.environments.sensors import CAR

    cfg = SensorConfig(
        name="ol", ray_angles=CAR.ray_angles, odometry=True, odom_source="commands", gyro=False
    )
    w = World(agent=AgentState(x=7.4, y=10.0, heading=0.0), walls=[Wall(8.0, 0.0, 1.0, 20.0)])
    suite = SensorSuite(cfg)
    suite.reset(w)
    for _ in range(10):
        obs = w.step(1.0, 1.0)  # pushing into the wall: the body does not move
        ch = suite.read(w, obs)
    assert w.agent.x == pytest.approx(7.4, abs=0.15)
    assert ch["odom_x"] == pytest.approx(7.4 + 10 * w.speed), "open loop believes it drove on"
    assert ch["gyro_heading"] == ch["odom_heading"], "no IMU: the gyro channel is the odometry"


def test_car_binary_wall_sensor_and_camera_beacon() -> None:
    from broomworm.environments.sensors import CAR
    from broomworm.environments.simple_2d import Dock

    cfg = SensorConfig(
        name="cam",
        ray_angles=CAR.ray_angles,
        wall_sensor=True,
        wall_binary=0.75,
        beacon_fov=math.radians(30.0),
        beacon_range=5.0,
    )
    w = World(agent=AgentState(x=5.0, y=10.0, heading=0.0), walls=[Wall(0.0, 9.0, 20.0, 0.4)])
    w.dock = Dock(x=8.0, y=10.0)
    suite = SensorSuite(cfg)
    suite.reset(w)
    ch = suite.read(w, w.observe())
    assert ch["wall_right"] == 1.0, "wall 0.6 to the right: the IR module is on"
    assert ch["dock_front"] == pytest.approx(1.0 / 3.0), "marker straight ahead, 3 away"
    w.agent = AgentState(x=5.0, y=10.0, heading=math.pi / 2)  # marker now at -90: out of view
    ch = suite.read(w, w.observe())
    assert (ch["dock_left"], ch["dock_front"], ch["dock_right"]) == (0.0, 0.0, 0.0)
    w.agent = AgentState(x=5.0, y=10.0, heading=-math.radians(20.0))  # marker 20 deg left
    ch = suite.read(w, w.observe())
    assert ch["dock_left"] > 0.0
    w.agent = AgentState(x=1.0, y=10.0, heading=0.0)  # 7 away: beyond the marker range
    ch = suite.read(w, w.observe())
    assert ch["dock_front"] == 0.0
    w.agent = AgentState(x=5.0, y=11.0, heading=0.0)  # wall 1.6 to the right: module off
    ch = suite.read(w, w.observe())
    assert ch["wall_right"] == 0.0


def test_camera_beacon_cannot_see_through_walls() -> None:
    from broomworm.environments.sensors import CAR
    from broomworm.environments.simple_2d import Dock

    cfg = SensorConfig(
        name="cam", ray_angles=CAR.ray_angles, beacon_fov=math.radians(30.0), beacon_range=5.0
    )
    w = World(agent=AgentState(x=5.0, y=10.0, heading=0.0), walls=[Wall(6.5, 8.0, 0.4, 4.0)])
    w.dock = Dock(x=8.0, y=10.0)
    suite = SensorSuite(cfg)
    suite.reset(w)
    ch = suite.read(w, w.observe())
    assert (ch["dock_left"], ch["dock_front"], ch["dock_right"]) == (0.0, 0.0, 0.0), (
        "wall in between"
    )
    w.walls = []
    ch = suite.read(w, w.observe())
    assert ch["dock_front"] > 0.0
