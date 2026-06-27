from flask import Flask, Response, render_template_string
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
import threading
import time
import winsound

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 96
FATIGUE_THRESHOLD = 0.25
CONSEC_FRAMES_FOR_ALERT = 10

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

model = mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(torch.load("models/drowsiness_cnn.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

alarm_active = threading.Event()

def alarm_loop():
    while True:
        if alarm_active.is_set():
            winsound.Beep(1500, 400)
            time.sleep(0.15)
        else:
            time.sleep(0.05)

threading.Thread(target=alarm_loop, daemon=True).start()

camera = cv2.VideoCapture(0)
fatigue_streak = 0
debug_saved = False

def crop_face(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    if not results.detections:
        return None
    box = results.detections[0].location_data.relative_bounding_box
    x = max(0, int(box.xmin * w))
    y = max(0, int(box.ymin * h))
    bw = int(box.width * w)
    bh = int(box.height * h)
    # pad a bit around the box
    pad_x = int(bw * 0.2)
    pad_y = int(bh * 0.2)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def generate_frames():
    global fatigue_streak, debug_saved
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        result = crop_face(frame)

        if result is None:
            cv2.putText(frame, "No face detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            fatigue_streak = 0
            alarm_active.clear()
        else:
            face_crop, (x1, y1, x2, y2) = result
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_face)

            if not debug_saved:
                pil_img.resize((96, 96)).save("debug_face_crop.jpg")
                debug_saved = True
                print("Saved debug_face_crop.jpg")

            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)
                prob_fatigue = F.softmax(output, dim=1)[0, 1].item()

            is_fatigue_frame = prob_fatigue >= FATIGUE_THRESHOLD
            fatigue_streak = fatigue_streak + 1 if is_fatigue_frame else 0
            alert_triggered = fatigue_streak >= CONSEC_FRAMES_FOR_ALERT
            print(f"prob={prob_fatigue:.3f} streak={fatigue_streak} alert={alert_triggered}")

            if alert_triggered:
                alarm_active.set()
            else:
                alarm_active.clear()

            color = (0, 0, 255) if alert_triggered else (0, 255, 0)
            status = "DROWSY ALERT!" if alert_triggered else ("Fatigue signal" if is_fatigue_frame else "Alert")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Fatigue prob: {prob_fatigue:.3f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if alert_triggered:
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

PAGE = """
<!DOCTYPE html>
<html>
<head><title>AI Drowsiness Detection</title>
<style>
body { background: #111; color: #eee; font-family: sans-serif; text-align: center; padding-top: 30px; }
img { border: 3px solid #444; border-radius: 8px; }
h1 { color: #4ade80; }
</style>
</head>
<body>
<h1>AI Drowsiness Detection — Live Demo</h1>
<p>Real CNN model (MobileNetV3, trained on UTA-RLDD) — not rule-based</p>
<img src="{{ url_for('video_feed') }}" width="720">
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)