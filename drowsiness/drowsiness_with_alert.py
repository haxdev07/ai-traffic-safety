import cv2
import mediapipe as mp
import numpy as np
import winsound
import threading
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.21
DROWSY_CONSEC_FRAMES = 20

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_points, w, h):
    coords = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]
    p1, p2, p3, p4, p5, p6 = coords
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)

# Shared flag between main loop and alarm thread
alarm_active = threading.Event()

def alarm_loop():
    while True:
        if alarm_active.is_set():
            winsound.Beep(1500, 400)
            time.sleep(0.15)
        else:
            time.sleep(0.05)

threading.Thread(target=alarm_loop, daemon=True).start()

cap = cv2.VideoCapture(0)
closed_frame_count = 0
alert_triggered = False
blink_count = 0
prev_closed = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        is_closed = avg_ear < EAR_THRESHOLD

        if is_closed:
            closed_frame_count += 1
        else:
            if prev_closed and closed_frame_count < DROWSY_CONSEC_FRAMES:
                blink_count += 1
            closed_frame_count = 0
            alert_triggered = False
            alarm_active.clear()

        prev_closed = is_closed

        if closed_frame_count >= DROWSY_CONSEC_FRAMES:
            alert_triggered = True
            alarm_active.set()

        color = (0, 0, 255) if alert_triggered else (0, 255, 0)
        status = "DROWSY ALERT!" if alert_triggered else ("Eyes closed" if is_closed else "Alert")

        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Closed frames: {closed_frame_count}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (30, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if alert_triggered:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
    else:
        alarm_active.clear()

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()