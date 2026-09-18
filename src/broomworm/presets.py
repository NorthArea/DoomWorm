"""The machines this track drives, as sensor presets (Plan §6).

A preset describes *hardware* — how many rangefinders, how noisy, whether there
is a bumper, what the odometry is blind to — so it belongs to the track that
owns the machine, not to the shared platform. Importing this module registers
them, and from then on the platform finds them by name like any other.

    vacuum   five rangefinders, bumper, cliff sensors, wheel odometry
    noisy    the same machine with its numbers made worse, for robustness runs
    car      the ACEBOTT QD001: one ultrasonic on a servo, no bumper, no
             encoders, a K210 marker for the dock
"""

from __future__ import annotations

import math

from wormlab.environments.sensors import SensorConfig, register_preset

VACUUM = SensorConfig(
    name="vacuum",
    ray_angles=tuple(math.radians(a) for a in (60.0, 30.0, 0.0, -30.0, -60.0)),
    noise_sigma=0.05,
    dropout=0.02,
    delay=1,
    bumper=True,
    cliff=True,
    wall_sensor=True,
    odometry=True,
    odom_sigma=0.05,
    gyro_drift=0.002,
)
NOISY = SensorConfig(
    name="noisy",
    ray_angles=VACUUM.ray_angles,
    noise_sigma=0.15,
    dropout=0.05,
    delay=2,
    bumper=True,
    cliff=True,
    wall_sensor=True,
    odometry=True,
    odom_sigma=0.15,
    gyro_drift=0.01,
)
# A kit car (stage 22: ACEBOTT QD001 + QD003 with one or two spare sensors): one
# ultrasonic on a servo swept over three headings, an IR obstacle module on the
# right, line-tracking modules under the nose as cliff sensors, no bumper, no
# encoders, no IMU; the K210 camera recognises the dock marker within its view.
CAR = SensorConfig(
    body="mecanum",  # QD001: four mecanum wheels, so it can also move sideways
    name="car",
    ray_angles=tuple(math.radians(a) for a in (30.0, 0.0, -30.0)),
    ray_range=4.0,
    noise_sigma=0.05,
    dropout=0.05,
    delay=1,
    sweep=True,
    proximity_bumper=0.7,
    cliff=True,
    wall_sensor=True,
    wall_range=1.0,
    wall_binary=0.75,
    odometry=True,
    odom_sigma=0.1,
    odom_source="commands",
    gyro=False,
    beacon_fov=math.radians(30.0),
    beacon_range=5.0,
)

for _preset in (VACUUM, NOISY, CAR):
    register_preset(_preset)
