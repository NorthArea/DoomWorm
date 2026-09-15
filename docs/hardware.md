# Hardware (phase A3, Plan §20.5)

Stage 22 moves the chosen brain onto a machine. Only the Environment and the
adapters change; the brain is the same saved file, wrapped in the same
engineered layer (map, planner, needs arbitration, dock autopilot, bumper
reflex). This page is the contract between the simulator and the robot, and
the procedure that checks it. It is written before any hardware exists, so
every number under "assumed" is a working value to be replaced by a
measurement.

## What the robot must provide

The channels of the `vacuum` sensor preset (`environments/sensors.py`), in
physical units, once per environment tick:

| Channel(s) | Sensor | Simulator model | Assumed part |
|---|---|---|---|
| `range_0..4` | 5 rangefinders at +60, +30, 0, -30, -60 degrees | proximity `1 - d / 4 u`, 5 % relative noise, 2 % dropout, 1 tick delay | ToF (VL53L1X class) or ultrasonic, reach >= 1.3 m |
| `bumper_left/right` | two-zone front bumper | contact this tick, side by nearest ray | microswitches |
| `cliff_left/right` | two downward IR sensors, 0.7 u ahead at +-30 degrees | edge of a danger zone | IR reflectance pair |
| `wall_right` | right-side IR | proximity `1 - d / 1 u` | IR distance sensor, reach ~0.3 m |
| `odom_x/y/heading` | wheel encoders, dead reckoning | 5 % multiplicative noise per displacement | quadrature encoders on both wheels |
| `gyro_heading` | IMU yaw | random-walk bias 0.002 rad/tick | MPU-6050 class IMU |
| `battery` | state of charge in [0, 1] | drains 0.002/tick, 500 ticks full to empty | cell voltage -> SoC table |
| `dock_left/front/right` | IR beacon receivers, three sectors | exact bearing within 8 u, `>= 0.5` within 2 u | 3 IR receivers + dock emitter |
| (`charging`) | dock contacts | "battery rising" is what the layer uses | contact flag, informational |

Actuators: differential drive, wheel commands in [-1, 1] as a fraction of the
maximum wheel speed. Controller: ESP32 or Raspberry Pi running the JSON-lines
protocol below; the brain runs on the host (laptop) during A3 (Plan §20.5
"controller class Raspberry Pi / ESP32"; on-board inference is a later step).

## Units (assumed, not measured)

The simulator uses abstract world units (u) and ticks. `Calibration`
(`hardware/calibration.py`) is the one place they meet metres and seconds:

| Quantity | Simulator | Assumed physical value | Source of the assumption |
|---|---|---|---|
| world unit | 1 u | 0.33 m | body radius 0.5 u = 0.165 m, a 33 cm vacuum |
| tick | 1 | 0.1 s | 10 Hz control loop |
| max wheel speed | 0.2 u/tick | 0.66 m/s | follows from the two above; a Roomba drives ~0.3 m/s, so the real tick or scale will change |
| body radius | 0.5 u | 0.165 m | |
| wheel base | 1.0 u | 0.33 m | wider than a real 0.23 m base; affects turn rate only |
| rangefinder reach | 4 u | 1.32 m | |
| wall IR reach | 1 u | 0.33 m | |
| cliff probe | 0.7 u ahead | 0.23 m | |
| door width (apartment maps) | 2.4 u | 0.79 m | |
| dock beacon range | 8 u | 2.6 m | |

Every value is a `Calibration` field or derived from one; `doomworm drive
--unit-m` overrides the scale. Nothing in a saved brain depends on these
numbers: the brain sees normalised channels only.

## Wire protocol

One JSON object per line, host -> robot, then robot -> host, one exchange per
tick. Transport: TCP (`doomworm drive --link tcp --host --port`), or any
text stream (a serial port wrapped as a text file works without new
dependencies).

```text
host  -> {"cmd": "reset"}                                  zero odometry, reply with the frame at rest
host  -> {"cmd": "drive", "left": 0.6, "right": -0.6}      wheels for one tick, reply with the next frame
robot -> {"tick": 12, "ranges_m": [0.41, null, 0.9, 1.2, null], "bumper": [0, 0], "cliff": [0, 0],
          "wall_m": 0.12, "odom_m": [1.02, -0.3], "odom_rad": 0.52, "gyro_rad": 0.5,
          "battery": 0.83, "charging": 0, "dock": [0.0, 0.0, 0.0]}
```

`null` in `ranges_m` / `wall_m` means no echo (nothing within reach). The
host converts this with `Calibration.channels()` into exactly the channel
dict the simulator's suite emits, same keys, same order.
`hardware/fake_robot.py` is the reference implementation of the robot side
(it serves the protocol from the simulator); the firmware imitates it.

## Procedure (Plan §20.5 order)

1. **Manual control and sensor recording** (stage 22.1, software side done):
   `doomworm drive --teleop --record runs/drive/<name>.jsonl` drives the
   simulator (`--link sim`) or the machine (`--link tcp`) with `w a s d x`
   keys or `left right` pairs from stdin and records every tick: wheels,
   channels, the raw physical reading, and the true pose when known.
2. **Comparison against the simulator** (stage 22.2):
   `doomworm compare-log --log <real log> --seed <map>` replays the recorded
   wheel sequence in a simulator map that mirrors the real room and reports
   per-channel RMSE, bumper agreement and (sim vs sim) the final pose error.
   The stage-22.1 demo shows the tool on two simulator logs: same preset ->
   zero error; a different preset -> the noise and geometry differences appear.
   Compare logs of the same preset only: ray indices differ between presets.
3. **Obstacle avoidance** (22.3): the A2 brain over `--link tcp` with
   `--planner needs`, in a room with walls and furniture.
4. **Dock** (22.4): beacon homing and charging on the real dock.
5. **Cleaning, map, call** (22.5): the acceptance of Plan §20.5.

## Commands

```bash
make demo-22                                   # brain over the sim link + log replay comparison
printf 'w\nw\nd\nw\n' | uv run doomworm drive --teleop --seed 3002 --record runs/drive/teleop.jsonl
uv run doomworm drive --teleop --link tcp --host 192.168.4.1 --port 5000 --record runs/drive/real.jsonl
uv run doomworm compare-log --log runs/drive/real.jsonl --seed 3002
uv run doomworm compare-log --log a.jsonl --against b.jsonl
```

The demo log and its report from the day the stage was written are under
`docs/results/stage22_drive_2026-09-15/`.
