# Firmware for the ACEBOTT QD001 car (stage 22.2)

`esp32_car/esp32_car.ino` is the robot side of the JSON-lines protocol in
`docs/hardware.md`: the ESP32 opens a Wi-Fi access point (`broomworm` /
`broomworm123`, address 192.168.4.1), listens on TCP port 5000, and answers
every `reset` / `drive` command with one sensor reading in physical units.
The brain and the planner layer run on the laptop:

```bash
uv run broomworm drive --link tcp --sensors car --teleop --record runs/drive/real.jsonl
uv run broomworm drive --link tcp --sensors car --brain runs/a2_car/<best>.json --planner needs
```

Status: **compiles** against the Arduino core for ESP32 (2.x and 3.x) with a
project-local `arduino-cli` (`make firmware`); **not run on hardware**.
Board: ACEBOTT ESP32 Max V1.0 = "ESP32 Dev Module" (FQBN `esp32:esp32:esp32`,
CH340 USB). Shield: QA052 "ESP32 Car Shield V1.0": motor directions through a
shift register (SHCP 18, STCP 17, DATA 5, EN 16), speed on PWM pin 19.

Flashing: the Max V1.0 has no BOOT button. Connect the pin labelled "00"
(GPIO0) to GND, press RST, `make firmware-flash PORT=/dev/cu.usbserial-XXXX`,
remove the jumper, press RST again.

Before the first drive:

1. Establish the shift-register bit map (`MOTOR_FWD` / `MOTOR_BWD`):
   `broomworm motor-map --link tcp` energises each bit alone and asks which
   wheel turned; paste the two lines it prints into the sketch and re-flash. Confirm the EN polarity and
   whether the shield has a second PWM pin (`PIN_PWM2`).
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
