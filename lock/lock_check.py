import discord
import cv2
import numpy as np
import time
import json
import os

# --- 1. 即時存檔邏輯 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "lock_config.json")
RESULT_IMG = os.path.join(BASE_DIR, "result.jpg")

def save_config_instant(low, up):
    with open(CONFIG_FILE, "w") as f:
        json.dump({"low": low.tolist(), "up": up.tolist()}, f)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            return np.array(data["low"]), np.array(data["up"])
    return np.array([25, 54, 171]), np.array([48, 110, 197]) # 預設

# --- 2. 核心裁切算法 (30% 放大) ---
ZOOM = 0.3 

def process_roi_logic(frame):
    h, w, _ = frame.shape
    rh, rw = int(h * ZOOM), int(w * ZOOM)
    sy, sx = (h - rh) // 2, (w - rw) // 2
    roi = frame[sy:sy+rh, sx:sx+rw]
    return roi, sy, sx, rh, rw

# --- 3. 調校模式 (改為即時自動存檔) ---
def run_tuner():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Tuner_Window")
    def nothing(x): pass
    
    low_init, up_init = load_config()
    cv2.createTrackbar("L-H", "Tuner_Window", low_init[0], 179, nothing)
    cv2.createTrackbar("U-H", "Tuner_Window", up_init[0], 179, nothing)
    cv2.createTrackbar("L-S", "Tuner_Window", low_init[1], 255, nothing)
    cv2.createTrackbar("U-S", "Tuner_Window", up_init[1], 255, nothing)
    cv2.createTrackbar("L-V", "Tuner_Window", low_init[2], 255, nothing)
    cv2.createTrackbar("U-V", "Tuner_Window", up_init[2], 255, nothing)

    print("--- 調校模式：滑桿變動即自動存檔 ---")
    print("對著視窗按 ESC 鍵即可進入 Discord 機器人模式")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        roi, _, _, _, _ = process_roi_logic(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 讀取並「即時」存檔
        l = np.array([cv2.getTrackbarPos("L-H", "Tuner_Window"), 
                      cv2.getTrackbarPos("L-S", "Tuner_Window"), 
                      cv2.getTrackbarPos("L-V", "Tuner_Window")])
        u = np.array([cv2.getTrackbarPos("U-H", "Tuner_Window"), 
                      cv2.getTrackbarPos("U-S", "Tuner_Window"), 
                      cv2.getTrackbarPos("U-V", "Tuner_Window")])
        save_config_instant(l, u)
        
        mask = cv2.inRange(hsv, l, u)
        res = cv2.bitwise_and(roi, roi, mask=mask)

        cv2.imshow("Tuner_Window", res)
        if cv2.waitKey(1) == 27: # 按 ESC 退出
            break
    cap.release()
    cv2.destroyAllWindows()

# --- 4. Discord 機器人主體 ---
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_message(message):
    if message.author == client.user: return

    if message.content == "!c":
        msg = await message.channel.send("...checking...")
        low, up = load_config() # 確保讀取到最新存檔
        
        cap = cv2.VideoCapture(0)
        time.sleep(1)
        ret, frame = cap.read()
        cap.release()

        if not ret: return

        roi, sy, sx, rh, rw = process_roi_logic(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low, up)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 核心視覺化：畫出紅色偵測邊框 (這框在哪，就代表機器人只看哪)
        cv2.rectangle(frame, (sx, sy), (sx + rw, sy + rh), (0, 0, 255), 3)
        
        res_text = "STATUS: UNDETECTED"
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 300: # 稍微調低門檻增加靈敏度
                bx, by, bw, bh = cv2.boundingRect(c)
                # 座標精準對齊：大圖 X = 小圖 x + 起始點 sx
                rx, ry = bx + sx, by + sy
                res_text = "STATUS: LOCKED" if bh > bw else "STATUS: UNLOCKED"
                cv2.rectangle(frame, (rx, ry), (rx + bw, ry + bh), (0, 255, 0), 2)
                cv2.putText(frame, res_text, (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(RESULT_IMG, frame)
        await msg.delete()
        await message.channel.send(res_text, file=discord.File(RESULT_IMG))

    elif message.content.startswith("!clear"):
        await message.channel.purge(limit=100)

if __name__ == "__main__":
    mode = input("是否啟動調校模式？(y/n): ")
    if mode.lower() == 'y': run_tuner()
    client.run('DISCORD_TOKEN')    