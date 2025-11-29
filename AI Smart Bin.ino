#include <Servo.h>
#include <DHT.h>

#define DHTPIN 2        // 温湿度传感器
#define DHTTYPE DHT11
#define IR_PIN 4        // 红外传感器

Servo myServo;
DHT dht(DHTPIN, DHTTYPE);

String inputString = "";
boolean stringComplete = false;
unsigned long lastSensorTime = 0;

// 定义三种状态
enum LidState {
  STATE_CLOSED,      // 状态0: 关闭
  STATE_OPEN_AUTO,   // 状态1: 自动模式 (会倒计时关闭)
  STATE_OPEN_HOLD    // 状态2: 投篮模式 (一直开，不倒计时)
};

LidState currentState = STATE_CLOSED; // 当前状态
unsigned long lidOpenStartTime = 0;   // 记录什么时候开盖的
const long AUTO_CLOSE_DELAY = 5000;   // 自动关盖时间 (5秒)


bool objectDetected = false; 

void setup() {
  Serial.begin(9600);
  myServo.attach(9);
  myServo.write(0); // 初始关闭
  dht.begin();
  pinMode(IR_PIN, INPUT); 
}

void loop() {
  // 1：处理串口指令 
  if (stringComplete) {
    inputString.trim();
    
    // 指令：普通自动开盖 (当AI 识别到物体)
    if (inputString == "OPEN_A") {
      // 只有在“非投篮模式”下，才允许 AI 控制盖子
      // 如果当前已经是 OPEN_HOLD，直接忽略 AI 的信号，防止干扰
      if (currentState != STATE_OPEN_HOLD) {
        currentState = STATE_OPEN_AUTO; // 进入倒计时模式（30秒）
        lidOpenStartTime = millis();    // 刷新计时器
        myServo.write(90);              // 开盖
      }
    }
    // 指令：投篮模式 (一直开)
    else if (inputString == "OPEN_HOLD") {
      currentState = STATE_OPEN_HOLD;   // 进入无限开盖模式
      myServo.write(90);                // 确保盖子是开的
      
    }
    // 指令：强制关闭 (超时或进球)
    else if (inputString == "CLOSE") {
      currentState = STATE_CLOSED;      // 切换回关闭状态
      myServo.write(0);                 // 关盖
    }
    
    inputString = "";
    stringComplete = false;
  }

  // 2：检查是否需要自动关盖
  // 只有在【自动模式】下才检查时间
  if (currentState == STATE_OPEN_AUTO) {
    // 如果时间超过了 5000 毫秒
    if (millis() - lidOpenStartTime >= AUTO_CLOSE_DELAY) {
      currentState = STATE_CLOSED; // 状态变回关闭
      myServo.write(0);            // 执行关盖
    }
  }


  // 3：红外线进球检测
  int irState = digitalRead(IR_PIN);
  if (irState == LOW && !objectDetected) {
    Serial.println("GOAL"); 
    objectDetected = true;
    delay(200); 
  } else if (irState == HIGH) {
    objectDetected = false; 
  }

  // 4：温湿度读取
  unsigned long currentMillis = millis();
  if (currentMillis - lastSensorTime >= 2000) {
    lastSensorTime = currentMillis;
    float h = dht.readHumidity();
    float t = dht.readTemperature();
    if (!isnan(h) && !isnan(t)) {
       Serial.print("TEMP: "); Serial.print((int)t);
       Serial.print("C | HUM: "); Serial.print((int)h);
       Serial.println("%");
    }
  }
}

// 串口中断
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;
    if (inChar == '\n') stringComplete = true;
  }
}