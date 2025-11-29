from flask import Flask, render_template, Response, jsonify
import torch
import cv2
import serial
import time
import numpy as np
from torchvision import models, transforms
from PIL import Image

# ================= ⚙️ 配置区域 =================
SERIAL_PORT = '/dev/cu.usbmodem1301' # ⚠️ 请确保端口正确
BAUD_RATE = 9600
MODEL_PATH = 'trash_model.pth'
CLASS_FILE = 'classes.txt'

CONFIDENCE_THRESHOLD = 0.60 
COOLDOWN_TIME = 0.5         
PROCESS_EVERY_N_FRAMES = 3 
GAME_TIMEOUT = 30.0 # 投篮限时 30 秒

app = Flask(__name__)
last_action_time = 0
ser = None

# 全局状态
current_status = {
    "class_name": "Waiting...",
    "confidence": 0,
    "last_command": "None",
    "sensor_data": "Connecting...",
    "game_mode": False,   
    "game_score": False   
}

game_start_time = 0

# --- 初始化串口 ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
    print(f"✅ Serial Connected: {SERIAL_PORT}")
except:
    print("⚠️ Serial Failed. Running in Simulation Mode.")

# --- 加载 AI 模型 ---
try:
    class_names = [line.strip() for line in open(CLASS_FILE)]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("🧠 Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    exit()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/start_game_mode', methods=['POST'])
def start_game_mode():
    """前端点击按钮后触发"""
    global game_start_time
    current_status["game_mode"] = True
    current_status["game_score"] = False
    game_start_time = time.time()
    
    # 发送指令：一直开盖！
    if ser and ser.is_open:
        ser.write(b"OPEN_HOLD\n")
        print("🏀 Game On: Lid OPEN_HOLD sent")
        
    return jsonify({"status": "started"})

def generate_frames():
    global last_action_time, current_status
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 目标物品列表 (只有这些能触发开盖)
    TARGET_OPEN_ITEMS = ["paper", "plastic"]
    OTHER_ITEMS = ["trash", "metal", "glass", "cardboard"]

    frame_counter = 0 
    score_display_start_time = 0 

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        frame_counter += 1

        # ================= 🎮 游戏模式逻辑 =================
        if current_status["game_mode"]:
            # 1. 检查超时
            if time.time() - game_start_time > GAME_TIMEOUT:
                current_status["game_mode"] = False
                if ser and ser.is_open: ser.write(b"CLOSE\n") # 超时关盖
                print("⏰ Timeout: Lid Closed")
            
            # 显示倒计时
            time_left = int(GAME_TIMEOUT - (time.time() - game_start_time))
            cv2.putText(frame, f"SHOOTING MODE: {time_left}s", (160, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 2. 自动重置进球显示状态 (3秒后)
        if current_status["game_score"] and (time.time() - score_display_start_time > 3.0):
            current_status["game_score"] = False
            print("🔄 Score Display Reset")

        # ================= 👁 AI 识别逻辑 =================
        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)

            conf, index = torch.max(probs, 0)
            class_name = class_names[index]
            score = conf.item()

            current_status["class_name"] = class_name
            current_status["confidence"] = int(score * 100)

            # ================= 🤖 自动控制逻辑 =================
            # 关键点：加了 `not current_status["game_mode"]` 判断
            # 意思是：如果正在玩游戏，绝对不要执行下面的自动开盖代码！
            if score > CONFIDENCE_THRESHOLD and not current_status["game_mode"]:
                command_to_send = None
                
                if class_name in TARGET_OPEN_ITEMS:
                    command_to_send = "OPEN_A" 
                
                if command_to_send:
                    current_time = time.time()
                    if current_time - last_action_time > COOLDOWN_TIME:
                        if ser and ser.is_open:
                            ser.write(f"{command_to_send}\n".encode('utf-8'))
                            print(f"🚀 AI Auto Command: {command_to_send}")
                        current_status["last_command"] = command_to_send
                        last_action_time = current_time

        # ================= 📡 串口读取 (检测进球) =================
        if ser and ser.in_waiting:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if line == "GOAL":
                    print("🎉 GOAL DETECTED!")
                    if current_status["game_mode"]:
                        current_status["game_score"] = True 
                        score_display_start_time = time.time()
                        
                        # 进球了！立刻结束游戏并关盖
                        current_status["game_mode"] = False 
                        if ser: ser.write(b"CLOSE\n")      
                elif line:
                    current_status["sensor_data"] = line
            except:
                pass

        # ================= 🎨 UI 绘制 =================
        display_name = current_status["class_name"]
        
        box_color = (0, 0, 255) 
        if display_name in TARGET_OPEN_ITEMS: 
            box_color = (0, 255, 0)
        elif display_name in OTHER_ITEMS: 
            box_color = (0, 165, 255)

        if time.time() - last_action_time > 1.0:
             current_status["last_command"] = "None"

        cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"{display_name}: {current_status['confidence']}%", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status(): return jsonify(current_status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)