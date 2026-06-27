import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
import winsound
import threading
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 96
FATIGUE_THRESHOLD = 0.25
CONSEC_FRAMES_FOR_ALERT = 10  # smooth over ~10 frames to avoid single-frame flicker

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
print("Model loaded. Classes: active=0, fatigue=1")

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
fatigue_streak = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        prob_fatigue = F.softmax(output, dim=1)[0, 1].item()

    is_fatigue_frame = prob_fatigue >= FATIGUE_THRESHOLD

    if is_fatigue_frame:
        fatigue_streak += 1
    else:
        fatigue_streak = 0

    alert_triggered = fatigue_streak >= CONSEC_FRAMES_FOR_ALERT

    if alert_triggered:
        alarm_active.set()
    else:
        alarm_active.clear()

    color = (0, 0, 255) if alert_triggered else (0, 255, 0)
    status = "DROWSY ALERT!" if alert_triggered else ("Fatigue signal" if is_fatigue_frame else "Alert")

    cv2.putText(frame, f"Fatigue prob: {prob_fatigue:.3f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, status, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Streak: {fatigue_streak}", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if alert_triggered:
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)

    cv2.imshow("ML Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()