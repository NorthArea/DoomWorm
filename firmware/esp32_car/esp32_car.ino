// BroomWorm stage 22.2: reference firmware for the ACEBOTT QD001 car (ESP32).
//
// Speaks the JSON-lines protocol of docs/hardware.md over TCP: the host sends
// {"cmd":"reset"} or {"cmd":"drive","left":l,"right":r} (l, r in [-1, 1]) and
// gets one RawReading back, in physical units. The Python side of this
// contract is broomworm/hardware/fake_robot.py; this sketch has to behave like it.
//
// STATUS: compiles against the Arduino core for ESP32 (2.x and 3.x), NOT run on
// hardware yet. Pins marked TODO come from the ACEBOTT documentation of the
// QA052 "ESP32 Car Shield V1.0" sample code, not from a measurement; the motor
// bit map of the shift register and every sensor pin must be confirmed on the
// bench (stage 23.2, docs/hardware.md, "day one").
//
// Board: ACEBOTT ESP32 Max V1.0 (Arduino IDE: "ESP32 Dev Module", FQBN
// esp32:esp32:esp32, CH340 USB). It has no BOOT button: to flash, connect the
// pin labelled "00" (GPIO0) to GND, press RST, upload, remove the jumper.
//
// Shield: QA052. Motors are driven through a shift register (74HC595-style:
// SHCP/STCP/DATA/EN) that sets the direction bits of all motors at once, plus a
// PWM pin for speed -- the same scheme as the old Adafruit motor shield v1.
// The documented sample uses SHCP 18, STCP 17, DATA 5, EN 16, PWM1 19 and the
// values 128 (forward) / 64 (backward) for one motor; which bit belongs to
// which wheel is TODO (see MOTOR_FWD / MOTOR_BWD and the bench procedure).
//
// Wiring assumed for the rest (change in the config block):
//   - 1 ultrasonic HC-SR04 on a servo, swept over 3 headings (+30, 0, -30 deg)
//   - 1 IR obstacle module on the right side (digital, LOW = obstacle)
//   - 3-way line-tracking module: outer left / right channels used as cliff sensors
//   - K210 module (QD003) on the shield's serial header: message format TODO,
//     until then dock = [0,0,0]
//   - no encoders (odom_m = null: the host integrates), no IMU
//
// Board libraries: WiFi.h and ESP32Servo. No JSON library: the two commands are
// parsed by hand, the reply is printed with snprintf.

#include <WiFi.h>
#include <ESP32Servo.h>

// ---------------------------------------------------------------- config (TODO: pins)
static const char* WIFI_SSID = "broomworm";      // the car runs its own access point
static const char* WIFI_PASS = "broomworm123";   // >= 8 chars for WPA2
static const uint16_t PORT = 5000;

static const int PIN_TRIG = 12;      // TODO: HC-SR04 trigger
static const int PIN_ECHO = 14;      // TODO: HC-SR04 echo (through a divider to 3.3 V)
static const int PIN_SERVO = 13;     // TODO: servo signal
static const int PIN_IR_RIGHT = 27;  // TODO: IR obstacle module OUT, right side
static const int PIN_LINE_L = 34;    // TODO: line module, outermost left channel
static const int PIN_LINE_R = 35;    // TODO: line module, outermost right channel
static const int PIN_BAT = 36;       // TODO: battery divider on an ADC pin, or -1 if none

// QA052 shield motor driver: shift register for directions + PWM for speed
// (pins from the ACEBOTT sample code; TODO: confirm on the bench).
static const int PIN_SR_CLOCK = 18;   // SHCP
static const int PIN_SR_LATCH = 17;   // STCP
static const int PIN_SR_DATA = 5;     // DATA
static const int PIN_SR_ENABLE = 16;  // EN (output enable, LOW = outputs on -- TODO: verify polarity)
static const int PIN_PWM1 = 19;       // speed of one motor group
static const int PIN_PWM2 = -1;       // TODO: the second PWM pin if the shield has one; -1 = share PWM1
static const int PWM_FREQ = 1000, PWM_BITS = 8;
static const int CH_L = 0, CH_R = 1;

// Shift-register bits per motor. The documented sample drives one motor with
// 128 (forward) and 64 (backward); the other three pairs are a guess to be
// established with the bench procedure in docs/hardware.md ("motor bit map"):
// send each bit alone, watch which wheel turns which way, fill the tables.
// Motor index: 0 front-left, 1 front-right, 2 rear-left, 3 rear-right.
static const uint8_t MOTOR_FWD[4] = {128, 32, 8, 2};   // TODO: measure
static const uint8_t MOTOR_BWD[4] = {64, 16, 4, 1};    // TODO: measure
static const int LEFT_MOTORS[2] = {0, 2};
static const int RIGHT_MOTORS[2] = {1, 3};

static const int SERVO_ANGLES[3] = {120, 90, 60};   // servo degrees for rays +30, 0, -30
static const float RANGE_MAX_M = 0.8f;              // 4 u * 0.2 m/u (Calibration.for_preset("car"))
static const unsigned long ECHO_TIMEOUT_US = 2UL * (unsigned long)(RANGE_MAX_M / 343.0f * 1e6f);
static const float BAT_FULL_V = 8.4f, BAT_EMPTY_V = 6.4f;   // 2S li-ion, TODO: measure
static const float BAT_DIVIDER = 3.0f;                       // TODO: your divider ratio

// ---------------------------------------------------------------- state
WiFiServer server(PORT);
Servo servo;
int sweepIndex = 0;
float ranges[3] = {-1.0f, -1.0f, -1.0f};   // < 0 = no echo (sent as null)
long tick = 0;
float lastLeft = 0.0f, lastRight = 0.0f;
int lastProbe = -1;                          // last bench probe pattern, echoed in the reply

// ---------------------------------------------------------------- motors
// The Arduino core for ESP32 changed its PWM API in 3.0: ledcAttach(pin, freq, bits)
// and ledcWrite(pin, duty) replaced ledcSetup/ledcAttachPin/ledcWrite(channel, duty).
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
static void pwmSetup(int pin, int /*channel*/) { if (pin >= 0) ledcAttach(pin, PWM_FREQ, PWM_BITS); }
static void pwmWrite(int pin, int /*channel*/, int duty) { if (pin >= 0) ledcWrite(pin, duty); }
#else
static void pwmSetup(int pin, int channel) {
  if (pin < 0) return;
  ledcSetup(channel, PWM_FREQ, PWM_BITS);
  ledcAttachPin(pin, channel);
}
static void pwmWrite(int pin, int channel, int duty) { if (pin >= 0) ledcWrite(channel, duty); }
#endif

static void writeShiftRegister(uint8_t bits) {
  digitalWrite(PIN_SR_LATCH, LOW);
  shiftOut(PIN_SR_DATA, PIN_SR_CLOCK, MSBFIRST, bits);
  digitalWrite(PIN_SR_LATCH, HIGH);
}

static uint8_t directionBits(const int motors[2], float cmd) {
  uint8_t bits = 0;
  if (fabsf(cmd) < 0.02f) return 0;               // brake: neither bit
  for (int k = 0; k < 2; k++) bits |= cmd > 0 ? MOTOR_FWD[motors[k]] : MOTOR_BWD[motors[k]];
  return bits;
}

static void drive(float left, float right) {
  left = constrain(left, -1.0f, 1.0f);
  right = constrain(right, -1.0f, 1.0f);
  lastLeft = left;
  lastRight = right;
  writeShiftRegister(directionBits(LEFT_MOTORS, left) | directionBits(RIGHT_MOTORS, right));
  const int full = (1 << PWM_BITS) - 1;
  if (PIN_PWM2 < 0) {
    // one PWM for every motor: the shared speed is the larger demand, the slower
    // side is only approximated (tank steering still works: opposite directions)
    pwmWrite(PIN_PWM1, CH_L, (int)(fmaxf(fabsf(left), fabsf(right)) * full));
  } else {
    pwmWrite(PIN_PWM1, CH_L, (int)(fabsf(left) * full));
    pwmWrite(PIN_PWM2, CH_R, (int)(fabsf(right) * full));
  }
}

// ---------------------------------------------------------------- sensors
static float pingMetres() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  unsigned long us = pulseIn(PIN_ECHO, HIGH, ECHO_TIMEOUT_US);
  if (us == 0) return -1.0f;                       // timeout = nothing within RANGE_MAX_M
  float m = us * 343.0f / 2.0f / 1e6f;
  return m > RANGE_MAX_M ? -1.0f : m;
}

// One servo position per tick, like SensorConfig(sweep=True) in the simulator.
static void sweepStep() {
  servo.write(SERVO_ANGLES[sweepIndex]);
  delay(15);                                       // TODO: measure the servo settle time
  ranges[sweepIndex] = pingMetres();
  sweepIndex = (sweepIndex + 1) % 3;
}

static float batteryFraction() {
  if (PIN_BAT < 0) return 1.0f;
  float v = analogReadMilliVolts(PIN_BAT) / 1000.0f * BAT_DIVIDER;
  return constrain((v - BAT_EMPTY_V) / (BAT_FULL_V - BAT_EMPTY_V), 0.0f, 1.0f);
}

// TODO: parse the K210 (QD003) serial output once its message format is known.
// Contract with the host: three sector strengths in [0, 1], left/front/right,
// 0 when the marker is not in view.
static void dockBeacon(float out[3]) {
  out[0] = out[1] = out[2] = 0.0f;
}

// ---------------------------------------------------------------- protocol
static void writeReading(WiFiClient& client) {
  char num[3][16];
  for (int i = 0; i < 3; i++) {
    if (ranges[i] < 0) snprintf(num[i], sizeof num[i], "null");
    else snprintf(num[i], sizeof num[i], "%.3f", ranges[i]);
  }
  float wallM = digitalRead(PIN_IR_RIGHT) == LOW ? 0.10f : -1.0f;   // binary module: 0.10 m or nothing
  int cliffL = digitalRead(PIN_LINE_L) == HIGH ? 1 : 0;              // TODO: polarity of your module
  int cliffR = digitalRead(PIN_LINE_R) == HIGH ? 1 : 0;
  float dock[3];
  dockBeacon(dock);
  char wall[16];
  if (wallM < 0) snprintf(wall, sizeof wall, "null");
  else snprintf(wall, sizeof wall, "%.3f", wallM);
  char probe[24] = "";
  if (lastProbe >= 0) snprintf(probe, sizeof probe, ",\"probe\":%d", lastProbe);
  char line[340];
  snprintf(line, sizeof line,
           "{\"tick\":%ld,\"ranges_m\":[%s,%s,%s],\"bumper\":[0,0],\"cliff\":[%d,%d],"
           "\"wall_m\":%s,\"odom_m\":null,\"odom_rad\":0,\"gyro_rad\":0,"
           "\"battery\":%.3f,\"charging\":0,\"dock\":[%.3f,%.3f,%.3f]%s}\n",
           tick, num[0], num[1], num[2], cliffL, cliffR, wall, batteryFraction(),
           dock[0], dock[1], dock[2], probe);
  client.print(line);
  lastProbe = -1;
}

static float numberAfter(const String& s, const char* key) {
  int i = s.indexOf(key);
  if (i < 0) return 0.0f;
  return s.substring(i + strlen(key)).toFloat();
}

static void handle(WiFiClient& client, const String& cmd) {
  if (cmd.indexOf("\"reset\"") >= 0) {
    drive(0.0f, 0.0f);
    tick = 0;
    for (int i = 0; i < 3; i++) sweepStep();     // a full sweep at rest
  } else if (cmd.indexOf("\"drive\"") >= 0) {
    drive(numberAfter(cmd, "\"left\":"), numberAfter(cmd, "\"right\":"));
    tick++;
    sweepStep();
  } else if (cmd.indexOf("\"probe\"") >= 0) {
    // Bench: energise one shift-register pattern for `ms` at `duty`, then stop
    // (broomworm motor-map builds MOTOR_FWD / MOTOR_BWD from what the operator sees).
    int bits = (int)numberAfter(cmd, "\"bits\":") & 0xFF;
    float duty = constrain(numberAfter(cmd, "\"duty\":"), 0.0f, 1.0f);
    int ms = constrain((int)numberAfter(cmd, "\"ms\":"), 50, 2000);
    const int full = (1 << PWM_BITS) - 1;
    writeShiftRegister((uint8_t)bits);
    pwmWrite(PIN_PWM1, CH_L, (int)(duty * full));
    pwmWrite(PIN_PWM2, CH_R, (int)(duty * full));
    delay(ms);
    drive(0.0f, 0.0f);
    lastProbe = bits;
  } else {
    // The host waits for exactly one line per command (LineLink._exchange blocks on
    // readline). Staying silent desyncs the link and hangs it until the socket
    // timeout, so say so and let the host fail loudly, like fake_robot.py does.
    client.print("{\"error\":\"unknown command\"}\n");
    drive(0.0f, 0.0f);
    return;
  }
  writeReading(client);
}

// ---------------------------------------------------------------- arduino
void setup() {
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_IR_RIGHT, INPUT);
  pinMode(PIN_LINE_L, INPUT);
  pinMode(PIN_LINE_R, INPUT);
  pinMode(PIN_SR_CLOCK, OUTPUT);
  pinMode(PIN_SR_LATCH, OUTPUT);
  pinMode(PIN_SR_DATA, OUTPUT);
  pinMode(PIN_SR_ENABLE, OUTPUT);
  digitalWrite(PIN_SR_ENABLE, LOW);              // TODO: verify (LOW = outputs enabled on a 595)
  pwmSetup(PIN_PWM1, CH_L);
  pwmSetup(PIN_PWM2, CH_R);
  servo.attach(PIN_SERVO);
  drive(0.0f, 0.0f);
  WiFi.softAP(WIFI_SSID, WIFI_PASS);             // host connects to 192.168.4.1:5000
  server.begin();
}

void loop() {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  WiFiClient client = server.accept();       // core 3.x renamed available()
#else
  WiFiClient client = server.available();
#endif
  if (!client) return;
  client.setTimeout(2000);
  unsigned long lastCommand = millis();
  while (client.connected()) {
    if (client.available()) {
      String cmd = client.readStringUntil('\n');
      lastCommand = millis();
      handle(client, cmd);
    } else {
      if (millis() - lastCommand > 500) {
        drive(0.0f, 0.0f);                       // watchdog: no host, no motion
      }
      delay(1);                                  // yield: a tight loop trips the task watchdog
    }
  }
  drive(0.0f, 0.0f);
}
