// DoomWorm stage 22.2: reference firmware for the ACEBOTT QD001 car (ESP32).
//
// Speaks the JSON-lines protocol of docs/hardware.md over TCP: the host sends
// {"cmd":"reset"} or {"cmd":"drive","left":l,"right":r} (l, r in [-1, 1]) and
// gets one RawReading back, in physical units. The Python side of this
// contract is doomworm/hardware/fake_robot.py; this sketch has to behave like it.
//
// STATUS: written before the kit was unboxed, NOT compiled, NOT run on hardware.
// Every pin and the motor-driver wiring below is a placeholder marked TODO;
// take them from the ACEBOTT tutorial PDF of your board revision.
//
// Wiring assumed (change in the config block):
//   - 1 ultrasonic HC-SR04 on a servo, swept over 3 headings (+30, 0, -30 deg)
//   - 1 IR obstacle module on the right side (digital, LOW = obstacle)
//   - line-tracking module: outer left / right channels used as cliff sensors
//   - 4 TT motors through the board's driver, left pair / right pair (tank steer)
//   - K210 module on a UART: sends "DOCK x_norm size\n" or nothing per frame (TODO: match
//     the QD003 firmware's actual message; until then dock = [0,0,0])
//   - no encoders (odom_m = null: the host integrates), no IMU
//
// Board libraries: WiFi.h and ESP32Servo (Arduino core for ESP32). No JSON library:
// the two commands are parsed by hand, the reply is printed with snprintf.

#include <WiFi.h>
#include <ESP32Servo.h>

// ---------------------------------------------------------------- config (TODO: pins)
static const char* WIFI_SSID = "doomworm";      // the car runs its own access point
static const char* WIFI_PASS = "doomworm123";   // >= 8 chars for WPA2
static const uint16_t PORT = 5000;

static const int PIN_TRIG = 12;      // TODO: HC-SR04 trigger
static const int PIN_ECHO = 14;      // TODO: HC-SR04 echo (through a divider to 3.3 V)
static const int PIN_SERVO = 13;     // TODO: servo signal
static const int PIN_IR_RIGHT = 27;  // TODO: IR obstacle module OUT, right side
static const int PIN_LINE_L = 34;    // TODO: line module, outermost left channel
static const int PIN_LINE_R = 35;    // TODO: line module, outermost right channel
static const int PIN_BAT = 36;       // TODO: battery divider on an ADC pin, or -1 if none

// TODO: motor driver. Two PWM + direction pairs (left pair, right pair) are assumed.
static const int PIN_L_IN1 = 16, PIN_L_IN2 = 17, PIN_L_PWM = 4;
static const int PIN_R_IN1 = 18, PIN_R_IN2 = 19, PIN_R_PWM = 5;
static const int PWM_FREQ = 1000, PWM_BITS = 8;
static const int CH_L = 0, CH_R = 1;

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

// ---------------------------------------------------------------- motors
static void motorPair(int in1, int in2, int ch, float cmd) {
  cmd = constrain(cmd, -1.0f, 1.0f);
  digitalWrite(in1, cmd >= 0 ? HIGH : LOW);
  digitalWrite(in2, cmd >= 0 ? LOW : HIGH);
  ledcWrite(ch, (int)(fabsf(cmd) * ((1 << PWM_BITS) - 1)));
}

static void drive(float left, float right) {
  lastLeft = left;
  lastRight = right;
  motorPair(PIN_L_IN1, PIN_L_IN2, CH_L, left);    // both left wheels
  motorPair(PIN_R_IN1, PIN_R_IN2, CH_R, right);   // both right wheels
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
  char line[320];
  snprintf(line, sizeof line,
           "{\"tick\":%ld,\"ranges_m\":[%s,%s,%s],\"bumper\":[0,0],\"cliff\":[%d,%d],"
           "\"wall_m\":%s,\"odom_m\":null,\"odom_rad\":0,\"gyro_rad\":0,"
           "\"battery\":%.3f,\"charging\":0,\"dock\":[%.3f,%.3f,%.3f]}\n",
           tick, num[0], num[1], num[2], cliffL, cliffR, wall, batteryFraction(),
           dock[0], dock[1], dock[2]);
  client.print(line);
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
  pinMode(PIN_L_IN1, OUTPUT); pinMode(PIN_L_IN2, OUTPUT);
  pinMode(PIN_R_IN1, OUTPUT); pinMode(PIN_R_IN2, OUTPUT);
  ledcSetup(CH_L, PWM_FREQ, PWM_BITS); ledcAttachPin(PIN_L_PWM, CH_L);
  ledcSetup(CH_R, PWM_FREQ, PWM_BITS); ledcAttachPin(PIN_R_PWM, CH_R);
  servo.attach(PIN_SERVO);
  drive(0.0f, 0.0f);
  WiFi.softAP(WIFI_SSID, WIFI_PASS);             // host connects to 192.168.4.1:5000
  server.begin();
}

void loop() {
  WiFiClient client = server.available();
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
