# Hardware (phase A3, Plan §20.5 and §20.5.1)

Stage 22 moves the chosen brain onto a machine. Only the Environment and the
adapters change; the brain is the same saved file, wrapped in the same
engineered layer (map, planner, needs arbitration, dock autopilot, bumper
reflex). This page is the contract between the simulator and the machine,
and the procedure that checks it. Every number under "assumed" is a working
value to be replaced by a measurement on the real car.

## The machine (decision 2026-09-15): a kit car, not a vacuum yet

ACEBOTT QD001 (ESP32, 4WD mecanum, one ultrasonic rangefinder, line-tracking
module, IR remote) with the QD003 expansion (K210 vision module) and one or
two basic sensors from Arduino starter kits. The simulator preset `car`
(`environments/sensors.py`) models exactly this, nothing more convenient:

| Channel(s) | On the car | Simulator model (`CAR`) | Assumed / to verify |
|---|---|---|---|
| `range_0..2` | one HC-SR04 on a servo, swept over +30 / 0 / -30 degrees | `sweep=True`: one ray refreshed per tick, the other two hold their last value; 5 % noise, 5 % dropout, 1 tick delay | the stock sensor sits on a servo (verify on unboxing; if fixed, one ray) |
| `bumper_left/right` | none | `proximity_bumper=0.7 u`: a ray closer than 0.7 u from the centre (4 cm from the body) reads as contact on its side; side collisions are invisible | |
| `cliff_left/right` | outer channels of the line-tracking module | edge of a danger zone 0.7 u ahead at +-30 degrees | polarity of the module |
| `wall_right` | IR obstacle module on the right side | binary: 1 within 0.75 u (5 cm from the body), else 0 | trim pot distance |
| `odom_x/y/heading` | no encoders | commanded wheels integrated (`odom_source="commands"`), 10 % noise; blind to slip and to pushing against a wall | mecanum slip is probably worse |
| `gyro_heading` | no IMU | equals the odometry heading | |
| `battery` | voltage divider if the board has one | the world's battery (0.002 per tick) | may be constant 1.0 on the real car |
| `dock_left/front/right` | K210 recognises a marker on the "dock" | visible within +-30 degrees and 5 u with line of sight; sector by bearing thirds; strength 1/d (apparent size) | K210 message format; the layer needs a distance estimate from the marker's size |
| (`charging`) | none: the "dock" is the start spot | rising battery = docked, in the simulator only | |

Actuators: four TT motors driven as two pairs (left, right) from wheel commands
in [-1, 1]; mecanum wheels under tank steering behave like a differential drive.

The camera is never an input of the brain (Plan §2.1, §34): it feeds the
engineered layer's dock beacon only.

## Units (assumed, not measured)

`Calibration.for_preset("car")` (`hardware/calibration.py`):

| Quantity | Simulator | Assumed physical value | Source of the assumption |
|---|---|---|---|
| world unit | 1 u | 0.20 m | body radius 0.5 u = 0.10 m, a ~20 cm car |
| tick | 1 | 0.1 s | 10 Hz loop; one servo position and one ping per tick |
| max wheel speed | 0.2 u/tick | 0.40 m/s | TT motors at 1:48 do roughly this; to measure |
| wheel base | 1.0 u | 0.20 m | |
| rangefinder reach | 4 u | 0.80 m | HC-SR04 does 4 m; capped so a ping fits in a tick |
| proximity bumper | 0.7 u | 0.14 m from the centre | |
| wall IR trip point | 0.75 u | 0.15 m from the centre | |
| door width (apartment maps) | 2.4 u | 0.48 m | narrow: a room with furniture, not a flat |
| marker recognition range | 5 u | 1.0 m | to measure with the K210 |

For the final vacuum platform the scale is 0.33 m per unit
(`Calibration.for_preset("vacuum")`); see the table at the end.

## Wire protocol

One JSON object per line, host -> robot, then robot -> host, one exchange per
tick. Transport: TCP (`doomworm drive --link tcp --host 192.168.4.1 --port 5000`,
the car is a Wi-Fi access point), or any text stream.

```text
host  -> {"cmd": "reset"}                                  stop, full sweep, reply with the frame at rest
host  -> {"cmd": "drive", "left": 0.6, "right": -0.6}      wheels for one tick, reply with the next frame
robot -> {"tick": 12, "ranges_m": [0.41, null, 0.30], "bumper": [0, 0], "cliff": [0, 0],
          "wall_m": 0.10, "odom_m": null, "odom_rad": 0, "gyro_rad": 0,
          "battery": 0.83, "charging": 0, "dock": [0.0, 0.0, 0.0]}
```

`null` in `ranges_m` / `wall_m` means no echo. `"odom_m": null` means the
machine has no encoders: the host integrates the commands it sent
(`LineLink._integrate_command`), the same dead reckoning the `car` preset
uses. `Calibration.channels()` turns the reading into exactly the channel
dict of the preset, same keys, same order.

Reference implementations: the robot side in Python is
`hardware/fake_robot.py` (serves the protocol from the simulator; the tests
drive a brain through it); the ESP32 sketch is `firmware/esp32_car/` (written
before unboxing, not compiled, pins are placeholders).

## Procedure (Plan §20.5 order)

1. **Manual control and sensor recording** (22.1, done on the simulator):
   `doomworm drive --teleop --sensors car --record runs/drive/<name>.jsonl`
   over `--link sim` or `--link tcp`.
2. **Comparison against the simulator** (22.2, needs the car): `doomworm
   compare-log --log <real log> --seed <map>` replays the recorded wheels in a
   simulator map that mirrors the room and reports per-channel RMSE, bumper
   agreement and (sim vs sim) the final pose error. Compare same-preset logs
   only: ray indices differ between presets.
3. **Obstacle avoidance** (22.3): the best `car`-preset brain over `--link tcp`
   with `--planner needs`, in a room with furniture.
4. **Marker "dock"** (22.4): K210 beacon into the layer, return to the start spot.
5. **Coverage, map, call** (22.5): the acceptance of Plan §20.5.1.

Brains are chosen on the `car` preset by the A2 protocol (`make benchmark-car`,
`make evolve-car CANDIDATE=...`); results in `docs/results/stage22_car_2026-09-15.md`.

## Day one on the car (everything below is ready, nothing needs the machine to prepare)

1. **Flash** `firmware/esp32_car/` after filling in the pins (firmware/README.md).
   Join the car's Wi-Fi (`doomworm` / `doomworm123`).
2. **Self-test**: `uv run doomworm selftest --link tcp --sensors car --out runs/selftest.md`.
   It checks the frame format, watches the sensors at rest (dropouts,
   round-trip latency), drives each wheel pair alone and forward, and says
   whether the odometry answers the right way round. Exit code 1 = a problem
   is listed at the end of the report.
3. **Calibrate**: `uv run doomworm calibrate --link tcp --sensors car --ticks 20`.
   Two runs, two tape-measure answers (metres driven, degrees turned) ->
   `runs/calibration.json` with the measured metres per unit, wheel base and
   tick length. Pass it to every later command with `--calibration runs/calibration.json`.
4. **Describe the room** in a file like `rooms/example_room.json` (metres:
   floor size, furniture rectangles, round obstacles, start pose, marker).
   The simulator builds the same room (`--room rooms/<name>.json` on any
   command; `build_world` accepts `room:<file>`) and the room travels inside
   every drive log, so a log replays without the file.
5. **First manual log**: `uv run doomworm drive --link tcp --sensors car --teleop
   --calibration runs/calibration.json --room rooms/<name>.json --record runs/drive/real1.jsonl`
   (keys `w a s d x`, `q` to stop). Then look at it: `uv run doomworm plot-log --log runs/drive/real1.jsonl`
   (path from odometry, bumper hits, rays, wheels).
6. **Compare with the simulator**: `uv run doomworm compare-log --log runs/drive/real1.jsonl`.
   Read the per-channel RMSE against the `car` preset: rays tell how far the
   noise/dropout numbers are off, odometry tells how much the command
   integration drifts. Adjust `CAR` in `environments/sensors.py` from the
   numbers, re-run `make benchmark-car`, and only then put a brain on the car.
7. **First autonomous drive**: the transferred curriculum worm
   (`--brain docs/results/brains_a2/worm_from_worm_evolved_random.json --planner needs`),
   then PPO-car (`docs/results/brains_car/ppo.json`, needs `uv sync --group rl`).

## Commands

```bash
make selftest-sim                              # the day-one self-test and calibration, on the simulator
make benchmark-car                             # every candidate on the car preset -> runs/benchmark_car/
make evolve-car CANDIDATE=rnn                  # retrain a candidate on the car preset -> runs/a2_car/
make demo-22                                   # brain over the sim link + log replay comparison
printf 'w\nw\nd\nw\n' | uv run doomworm drive --teleop --sensors car --seed 3002 --record runs/drive/teleop.jsonl
uv run doomworm drive --teleop --link tcp --sensors car --record runs/drive/real.jsonl
uv run doomworm compare-log --log runs/drive/real.jsonl --seed 3002
```

## The vacuum (the project's target platform, Plan §48)

Kept for when the real vacuum arrives; the `vacuum` preset and its A2 table
(`docs/results/a2_bakeoff_2026-09-15.md`) stay the reference.

| Channel(s) | Sensor | Assumed part |
|---|---|---|
| `range_0..4` | 5 rangefinders at +60, +30, 0, -30, -60 degrees, reach >= 1.3 m | ToF (VL53L1X class) or ultrasonic |
| `bumper_left/right` | two-zone front bumper | microswitches |
| `cliff_left/right` | two downward IR sensors 0.23 m ahead at +-30 degrees | IR reflectance pair |
| `wall_right` | right-side IR, reach ~0.3 m | IR distance sensor |
| `odom_*`, `gyro_heading` | wheel encoders + IMU yaw | quadrature encoders, MPU-6050 class |
| `battery`, `charging` | state of charge, dock contacts | cell voltage -> SoC |
| `dock_*` | three IR beacon receivers + dock emitter, range 2.6 m | |

Scale 0.33 m per unit (33 cm body), tick 0.1 s, rangefinder reach 1.32 m,
door width 0.79 m; the implied 0.66 m/s top speed is above a real vacuum's.
A used Roomba 500-800 / Create with the Open Interface serial port provides
all of this except the rangefinders.
