# Firmware for the ACEBOTT QD001 car (stage 22.2)

`esp32_car/esp32_car.ino` is the robot side of the JSON-lines protocol in
`docs/hardware.md`: the ESP32 opens a Wi-Fi access point (`doomworm` /
`doomworm123`, address 192.168.4.1), listens on TCP port 5000, and answers
every `reset` / `drive` command with one sensor reading in physical units.
The brain and the planner layer run on the laptop:

```bash
uv run doomworm drive --link tcp --sensors car --teleop --record runs/drive/real.jsonl
uv run doomworm drive --link tcp --sensors car --brain runs/a2_car/<best>.json --planner needs
```

Status: written before the kit was unboxed. **Not compiled, not run on
hardware.** Before flashing:

1. Take every `PIN_*` and the motor-driver wiring from the ACEBOTT tutorial
   PDF of your board revision (the `TODO` block at the top of the sketch).
2. Check the polarity of the line-tracking module (HIGH or LOW on a dark /
   missing floor) and the trip distance of the IR obstacle module.
3. Measure the servo settle time and the real top speed; put the speed into
   `Calibration.for_preset("car")` (`speed_mps` is 0.40 m/s today, assumed).
4. Replace `dockBeacon()` with a parser of the QD003 (K210) serial message
   once its format is known; until then the dock channels are zeros.

Behaviour to keep identical to `hardware/fake_robot.py` and the `car` preset:
one servo position and one ping per tick, `null` for no echo, `odom_m: null`
(the host integrates the commanded wheels), motors stop if no command arrives
for 500 ms.

Libraries: the Arduino core for ESP32 (`WiFi.h`) and `ESP32Servo`. No JSON
library; the two commands are parsed by hand.
