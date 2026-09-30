#include <WiFi.h>
#include "hotspot_wifi_config.h"

WiFiServer server(80);
unsigned long lastWiFiAttempt = 0;
// Car Wi-Fi stays available whether hotspot Wi-Fi connects or not.
const char* CAR_WIFI_SSID = "SkyRobot";
const char* CAR_WIFI_PASSWORD = "SkyRobot32";
bool hotspotAttemptActive = true;
bool hotspotWasConnected = false;

void serviceWiFi() {
  bool connected = WiFi.status() == WL_CONNECTED;
  if (connected && !hotspotWasConnected) {
    Serial.print("Hotspot IP: "); Serial.println(WiFi.localIP());
  }
  if (connected) hotspotAttemptActive = false;
  // Stop hotspot connection attempt once a laptop joins the car, or after 15 seconds.
  // Reboot with a hotspot to try hotspot Wi-Fi again.
  if (!connected && (hotspotWasConnected ||
      (hotspotAttemptActive && (WiFi.softAPgetStationNum() > 0 ||
                              millis() - lastWiFiAttempt >= 15000)))) {
    WiFi.disconnect(false, false);  // Keep the car access point running.
    hotspotAttemptActive = false;
    Serial.println("Hotspot unavailable; use SkyRobot Wi-Fi at 192.168.4.1");
  }
  hotspotWasConnected = connected;
}

#define M1_IN1 27
#define M1_IN2 26
#define M2_IN1 16
#define M2_IN2 17

// Two positional servos, mirrored mounting assumed. Arduino-ESP32 3.x.
// These are example angles, NOT calibrated mechanical limits.
// Calibrate each arm with its linkage unloaded before gripping an object.
constexpr int GRIP_LEFT_PIN = 32;
constexpr int GRIP_RIGHT_PIN = 33;
int GRIP_LEFT_OPEN_ANGLE = 90;
int GRIP_LEFT_CLOSE_ANGLE = 150;
int GRIP_RIGHT_OPEN_ANGLE = 90;
int GRIP_RIGHT_CLOSE_ANGLE = 30;
constexpr int SERVO_MIN_US = 1000;
constexpr int SERVO_MAX_US = 2000;
bool gripperReady = false;
char gripState = '?';

unsigned long gripDuty(int angle) {
  angle = constrain(angle, 0, 180);
  unsigned long pulse = map(angle, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  return (pulse * 65535UL + 10000UL) / 20000UL;
}

void setGripper(bool open) {
  if (!gripperReady) return;
  char target = open ? 'O' : 'C';
  if (target == gripState) return;
  // Separate writes: always attempt BOTH outputs.
  bool leftOK = ledcWrite(GRIP_LEFT_PIN,
      gripDuty(open ? GRIP_LEFT_OPEN_ANGLE : GRIP_LEFT_CLOSE_ANGLE));
  bool rightOK = ledcWrite(GRIP_RIGHT_PIN,
      gripDuty(open ? GRIP_RIGHT_OPEN_ANGLE : GRIP_RIGHT_CLOSE_ANGLE));
  if (leftOK && rightOK) {
    gripState = target;
    Serial.println(open ? "Both gripper arms OPEN command" : "Both gripper arms CLOSE command");
  } else {
    gripState = '?';
    Serial.println("ERROR: gripper PWM write failed");
  }
}

int m1Speed = 227;
int m2Speed = 245;
int leftInnerSpeed = 210;
int rightInnerSpeed = 210;
char motion = 'S';
unsigned long lastMotionCommand = 0;
unsigned long pulseStopAt = 0;
bool pulseActive = false;

void stopCar() {
  ledcWrite(M1_IN1, 0); ledcWrite(M1_IN2, 0);
  ledcWrite(M2_IN1, 0); ledcWrite(M2_IN2, 0);
}

void serviceTimedPulse() {
  if (pulseActive && (long)(millis() - pulseStopAt) >= 0) {
    pulseActive = false;
    motion = 'S';
    stopCar();
  }
}

void applyMotion() {
  stopCar();
  switch (motion) {
    case 'F':
      ledcWrite(M1_IN1, m1Speed); ledcWrite(M2_IN1, m2Speed); break;
    case 'B':
      ledcWrite(M1_IN2, m1Speed); ledcWrite(M2_IN2, m2Speed); break;
    case 'L':
      // M1 rotates backward, M2 rotates forward: spin left in place.
      ledcWrite(M1_IN2, leftInnerSpeed); ledcWrite(M2_IN1, rightInnerSpeed); break;
    case 'R':
      // M1 rotates forward, M2 rotates backward: spin right in place.
      ledcWrite(M1_IN1, leftInnerSpeed); ledcWrite(M2_IN2, rightInnerSpeed); break;
  }
}

void setup() {
  Serial.begin(115200);
  ledcAttachChannel(M1_IN1, 20000, 8, 0);
  ledcAttachChannel(M1_IN2, 20000, 8, 1);
  ledcAttachChannel(M2_IN1, 20000, 8, 2);
  ledcAttachChannel(M2_IN2, 20000, 8, 3);
  stopCar();
  // Channels 6/7 share a 50 Hz timer but have independent duties.
  // Motor channels 0-3 use separate timers on classic ESP32.
  bool leftReady = ledcAttachChannel(GRIP_LEFT_PIN, 50, 16, 6);
  bool rightReady = ledcAttachChannel(GRIP_RIGHT_PIN, 50, 16, 7);
  if (leftReady) ledcWrite(GRIP_LEFT_PIN, 0);
  if (rightReady) ledcWrite(GRIP_RIGHT_PIN, 0);
  gripperReady = leftReady && rightReady;
  // No startup position command; wait for a gesture.
  if (!gripperReady) {
    Serial.println("ERROR: gripper PWM attach failed; both arms disabled");
  }
  WiFi.mode(WIFI_AP_STA);
  WiFi.setAutoReconnect(false);
  IPAddress carIP(192, 168, 4, 1);
  IPAddress subnet(255, 255, 255, 0);
  bool apConfigured = WiFi.softAPConfig(carIP, carIP, subnet);
  bool apStarted = apConfigured && WiFi.softAP(CAR_WIFI_SSID, CAR_WIFI_PASSWORD);
  if (apStarted) {
    Serial.print("Car Wi-Fi: "); Serial.println(CAR_WIFI_SSID);
    Serial.print("Car IP: "); Serial.println(WiFi.softAPIP());
  } else {
    Serial.println("ERROR: car Wi-Fi could not start");
  }
  server.begin();
  // Mobile hotspot uses ordinary Wi-Fi password authentication.
  WiFi.begin(WIFI_STA_SSID, WIFI_STA_PASSWORD);
  lastWiFiAttempt = millis();
  Serial.println("Connecting to hotspot Wi-Fi...");
}

void loop() {
  serviceWiFi();
  serviceTimedPulse();
  if (motion != 'S' && millis() - lastMotionCommand > 700) {
    motion = 'S';
    pulseActive = false;
    stopCar();
  }
  WiFiClient client = server.available();
  if (!client) return;
  String settingLine;
  bool readingSettings = false;
  char settingsType = 'T';
  while (client.connected()) {
    serviceWiFi();
    serviceTimedPulse();
    if (motion != 'S' && millis() - lastMotionCommand > 700) {
      motion = 'S';
      pulseActive = false;
      stopCar();
    }
    if (!client.available()) { delay(2); continue; }
    char c = client.read();
    if (readingSettings) {
      if (c == '\n') {
        if (settingsType == 'P') {
          char direction, extra;
          unsigned long durationMs;
          if (sscanf(settingLine.c_str(), "%c,%lu%c", &direction, &durationMs, &extra) == 2 &&
              (direction == 'F' || direction == 'B' || direction == 'L' || direction == 'R') &&
              durationMs >= 20 && durationMs <= 500) {
            motion = direction;
            lastMotionCommand = millis();
            pulseStopAt = lastMotionCommand + durationMs;
            pulseActive = true;
            applyMotion();
          } else {
            motion = 'S';
            pulseActive = false;
            stopCar();
          }
          settingLine = "";
          readingSettings = false;
          continue;
        }
        int a, b, left, right;
        char extra;
        // A: left open, left close, right open, right close.
        if (settingsType == 'A') {
          motion = 'S';
          pulseActive = false;
          stopCar();
          if (sscanf(settingLine.c_str(), "%d,%d,%d,%d%c", &a, &b, &left, &right, &extra) == 4 &&
              a >= 0 && a <= 180 && b >= 0 && b <= 180 &&
              left >= 0 && left <= 180 && right >= 0 && right <= 180) {
            GRIP_LEFT_OPEN_ANGLE = a; GRIP_LEFT_CLOSE_ANGLE = b;
            GRIP_RIGHT_OPEN_ANGLE = left; GRIP_RIGHT_CLOSE_ANGLE = right;
            gripState = '?'; // Next open/close applies the new angles.
            client.print("OK\n"); // Save only; no servo movement here.
          } else {
            client.print("ERR\n");
          }
          settingLine = "";
          readingSettings = false;
          continue;
        }
        // Entire packet: T<four comma-separated PWM values>\n
        if (sscanf(settingLine.c_str(), "%d,%d,%d,%d%c", &a, &b, &left, &right, &extra) == 4 &&
            a >= 0 && a <= 255 && b >= 0 && b <= 255 &&
            left >= 0 && left <= 255 && right >= 0 && right <= 255) {
          m1Speed = a; m2Speed = b;
          leftInnerSpeed = left; rightInnerSpeed = right;
          pulseActive = false;
          applyMotion();
          client.print("OK\n");
        } else {
          client.print("ERR\n");
        }
        settingLine = "";
        readingSettings = false;
      } else if (settingLine.length() < 24) {
        settingLine += c;
      } else {
        settingLine = "";
        readingSettings = false;
        client.print("ERR\n");
      }
    } else if (c == 'Q') {
      client.printf("ANGLES,%d,%d,%d,%d\n", GRIP_LEFT_OPEN_ANGLE,
                    GRIP_LEFT_CLOSE_ANGLE, GRIP_RIGHT_OPEN_ANGLE, GRIP_RIGHT_CLOSE_ANGLE);
    } else if (c == 'T' || c == 'A' || c == 'P') {
      settingsType = c;
      readingSettings = true;
      settingLine = "";
    } else if (c == 'O' || c == 'C') {
      motion = 'S';
      pulseActive = false;
      stopCar();
      setGripper(c == 'O');
    } else if (c == 'F' || c == 'B' || c == 'L' || c == 'R' || c == 'S') {
      motion = c;
      pulseActive = false;
      lastMotionCommand = millis();
      applyMotion();
    }
  }
  motion = 'S';
  pulseActive = false;
  stopCar();
  client.stop();
}
