import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# 提高複雜度至 2 (最準確模式) 以減少誤判
pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # 取得關鍵點 (以右半身為例)
        r_shldr = [lm[12].x, lm[12].y]
        r_elbow = [lm[14].x, lm[14].y]
        r_hip = [lm[24].x, lm[24].y]
        
        # --- 進階邏輯：計算「肩外展角」(手臂與側邊軀幹的夾角) ---
        # 這種算法不會因為你身體前傾而誤判
        abduction_angle = get_angle(r_hip, r_shldr, r_elbow)
        
        # 專業判定：投球加速期手肘應與肩線維持在 80-100 度
        is_safe = 80 <= abduction_angle <= 110
        color = (0, 255, 0) if is_safe else (0, 0, 255)
        
        if not is_safe and lm[14].y > lm[12].y: # 雙重判定：角度不對且手肘確實低於肩膀
            cv2.putText(frame, "ELBOW COLLAPSE!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 顯示角度
        cv2.putText(frame, f"Abduction: {int(abduction_angle)} deg", 
                    (int(r_shldr[0]*frame.shape[1]), int(r_shldr[1]*frame.shape[0])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(255,255,255), thickness=2),
                                 mp_drawing.DrawingSpec(color=color, thickness=2))

    cv2.imshow('Advanced Pitching Analyst', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
# python3 pitch_check.py