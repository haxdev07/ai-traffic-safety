from flask import Flask, render_template_string, request
import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)

FEATURES_DIR = "data/crash_features"
IMAGES_DIR = "data/CrashBest"
DEVICE = torch.device("cpu")
THRESHOLD = 0.2

mean = np.load("models/accident_norm_mean.npy")
std = np.load("models/accident_norm_std.npy")

class AccidentLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

sample_file = sorted(os.listdir(FEATURES_DIR))[0]
sample_data = np.load(os.path.join(FEATURES_DIR, sample_file))
feature_dim = sample_data["features"].shape[1]

model = AccidentLSTM(input_dim=feature_dim).to(DEVICE)
model.load_state_dict(torch.load("models/accident_lstm.pth", map_location=DEVICE))
model.eval()

def list_videos():
    files = sorted(os.listdir(FEATURES_DIR))
    return [f.replace(".npz", "") for f in files]

def predict_video(vid_id):
    data = np.load(os.path.join(FEATURES_DIR, f"{vid_id}.npz"))
    features = (data["features"] - mean) / (std + 1e-6)
    labels = data["labels"]

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1)[0, :, 1].numpy()

    return probs, labels

def make_plot(vid_id, probs, labels):
    fig, ax = plt.subplots(figsize=(10, 3))
    frames = np.arange(1, 51)
    ax.plot(frames, probs, color="#f87171", linewidth=2, label="Predicted crash probability")
    ax.axhline(y=THRESHOLD, color="#facc15", linestyle="--", label=f"Decision threshold ({THRESHOLD})")

    actual_crash_frames = frames[labels == 1]
    if len(actual_crash_frames) > 0:
        ax.axvspan(actual_crash_frames[0], actual_crash_frames[-1], color="#f87171", alpha=0.15, label="Actual crash window")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Frame number (1-50, 5 second clip)")
    ax.set_ylabel("Crash probability")
    ax.set_title(f"Video {vid_id} — frame-by-frame crash prediction")
    ax.legend(loc="upper left", fontsize=8)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def get_sample_frame_b64(vid_id, frame_num):
    img_path = os.path.join(IMAGES_DIR, f"C_{vid_id}_{frame_num:02d}.jpg")
    if not os.path.exists(img_path):
        return None
    img = Image.open(img_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

PAGE = """
<!DOCTYPE html>
<html>
<head><title>Accident Detection Demo</title>
<style>
body { background: #111; color: #eee; font-family: sans-serif; padding: 30px; }
h1 { color: #f87171; }
select, button { font-size: 16px; padding: 8px; margin: 10px 5px 20px 0; }
img.plot { border: 2px solid #444; border-radius: 8px; margin-top: 10px; }
img.frame { border: 2px solid #444; border-radius: 4px; margin: 4px; width: 110px; }
.stats { background: #1c1c1c; padding: 15px; border-radius: 8px; margin-top: 15px; }
.frames-row { display: flex; flex-wrap: wrap; margin-top: 15px; }
</style>
</head>
<body>
<h1>AI Accident Detection — Live Demo</h1>
<p>Real CNN+LSTM model trained on CCD (90.7% crash recall at tuned threshold)</p>
<form method="get">
<select name="vid">
{% for v in videos %}
<option value="{{ v }}" {% if v == selected %}selected{% endif %}>{{ v }}</option>
{% endfor %}
</select>
<button type="submit">Run prediction</button>
</form>
{% if plot %}
<img class="plot" src="data:image/png;base64,{{ plot }}" width="900">
<div class="stats">
  <p>Actual crash frames: <b>{{ crash_frame_list }}</b></p>
  <p>Predicted crash frames (above threshold): <b>{{ pred_frame_list }}</b></p>
</div>
<h3>Sample frames around predicted crash point</h3>
<div class="frames-row">
{% for fnum, b64 in sample_frames %}
  <div>
    <img class="frame" src="data:image/jpeg;base64,{{ b64 }}">
    <p style="text-align:center; margin:2px; font-size:12px;">Frame {{ fnum }}</p>
  </div>
{% endfor %}
</div>
{% endif %}
</body>
</html>
"""

@app.route("/")
def index():
    videos = list_videos()
    selected = request.args.get("vid", videos[0])
    plot = None
    crash_frame_list = ""
    pred_frame_list = ""
    sample_frames = []

    if selected:
        probs, labels = predict_video(selected)
        plot = make_plot(selected, probs, labels)

        actual_crash_idx = np.where(labels == 1)[0] + 1
        pred_crash_idx = np.where(probs >= THRESHOLD)[0] + 1
        crash_frame_list = ", ".join(map(str, actual_crash_idx)) if len(actual_crash_idx) else "none"
        pred_frame_list = ", ".join(map(str, pred_crash_idx)) if len(pred_crash_idx) else "none"

        center = actual_crash_idx[0] if len(actual_crash_idx) else 25
        frame_range = range(max(1, center - 2), min(50, center + 3))
        for fnum in frame_range:
            b64 = get_sample_frame_b64(selected, fnum)
            if b64:
                sample_frames.append((fnum, b64))

    return render_template_string(PAGE, videos=videos, selected=selected, plot=plot,
                                   crash_frame_list=crash_frame_list, pred_frame_list=pred_frame_list,
                                   sample_frames=sample_frames)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5002)