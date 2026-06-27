from flask import Flask, render_template_string, request
import os
import re
import io
import base64
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

DATA_ROOT = "data/UAH-DRIVESET-v1"
WINDOW_SIZE = 30
STEP_SIZE = 10
DEVICE = torch.device("cpu")

COLUMNS = [
    "timestamp", "lat", "lon",
    "score_total_w", "score_acc_w", "score_brake_w", "score_turn_w",
    "score_weave_w", "score_drift_w", "score_speed_w", "score_follow_w",
    "ratio_normal_w", "ratio_drowsy_w", "ratio_aggr_w", "ratio_distract_w",
    "score_total", "score_acc", "score_brake", "score_turn",
    "score_weave", "score_drift", "score_speed", "score_follow",
    "ratio_normal", "ratio_drowsy", "ratio_aggr", "ratio_distract"
]
FEATURE_COLS = [
    "score_acc_w", "score_brake_w", "score_turn_w",
    "score_weave_w", "score_drift_w", "score_speed_w", "score_follow_w"
]
LABEL_NAMES = ["NORMAL", "DROWSY", "AGGRESSIVE"]
LABEL_COLORS = ["#4ade80", "#facc15", "#f87171"]

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model = LSTMClassifier(input_dim=len(FEATURE_COLS)).to(DEVICE)
model.load_state_dict(torch.load("models/behavior_lstm_final.pth", map_location=DEVICE))
model.eval()
mean = np.load("models/behavior_norm_mean.npy")
std = np.load("models/behavior_norm_std.npy")

def parse_behavior(folder_name):
    parts = folder_name.split("-")
    return re.sub(r"\d+$", "", parts[3])

def list_trips():
    trips = []
    for driver in sorted(os.listdir(DATA_ROOT)):
        driver_path = os.path.join(DATA_ROOT, driver)
        if not os.path.isdir(driver_path) or not driver.startswith("D"):
            continue
        for trip in sorted(os.listdir(driver_path)):
            trip_path = os.path.join(driver_path, trip)
            if os.path.exists(os.path.join(trip_path, "SEMANTIC_ONLINE.txt")):
                trips.append(f"{driver}/{trip}")
    return trips

def predict_trip(trip_id):
    semantic_file = os.path.join(DATA_ROOT, trip_id, "SEMANTIC_ONLINE.txt")
    df = pd.read_csv(semantic_file, sep=r"\s+", header=None, names=COLUMNS)
    features = df[FEATURE_COLS].values.astype(np.float32)
    timestamps = df["timestamp"].values

    actual_behavior = parse_behavior(trip_id.split("/")[1])

    preds = []
    times = []
    for start in range(0, len(features) - WINDOW_SIZE, STEP_SIZE):
        window = features[start:start + WINDOW_SIZE]
        norm_window = (window - mean) / (std + 1e-6)
        x = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)
            pred = out.argmax(1).item()
        preds.append(pred)
        times.append(timestamps[start + WINDOW_SIZE - 1])

    return times, preds, actual_behavior

def make_plot(times, preds, actual_behavior):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    colors = [LABEL_COLORS[p] for p in preds]
    ax.bar(times, [1] * len(times), width=(times[1]-times[0]) if len(times) > 1 else 10, color=colors)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds into trip)")
    ax.set_title(f"Predicted behavior over time  |  Actual label: {actual_behavior}")

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=LABEL_COLORS[i], label=LABEL_NAMES[i]) for i in range(3)]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

PAGE = """
<!DOCTYPE html>
<html>
<head><title>Driving Behavior Demo</title>
<style>
body { background: #111; color: #eee; font-family: sans-serif; padding: 30px; }
h1 { color: #4ade80; }
select, button { font-size: 16px; padding: 8px; margin: 10px 5px 20px 0; }
img { border: 2px solid #444; border-radius: 8px; margin-top: 10px; }
.stats { background: #1c1c1c; padding: 15px; border-radius: 8px; margin-top: 15px; }
.match { color: #4ade80; font-weight: bold; }
.mismatch { color: #f87171; font-weight: bold; }
</style>
</head>
<body>
<h1>AI Driving Behavior Classification — Live Demo</h1>
<p>Real LSTM model trained on UAH-DriveSet (86.6% mean cross-driver accuracy via leave-one-driver-out)</p>
<form method="get">
<select name="trip">
{% for t in trips %}
<option value="{{ t }}" {% if t == selected %}selected{% endif %}>{{ t }}</option>
{% endfor %}
</select>
<button type="submit">Run prediction</button>
</form>
{% if plot %}
<img src="data:image/png;base64,{{ plot }}" width="900">
<div class="stats">
  <p>Actual label: <b>{{ actual }}</b></p>
  <p>Predicted-window agreement with actual label:
    <span class="{{ 'match' if agreement >= 0.5 else 'mismatch' }}">{{ "%.1f"|format(agreement*100) }}%</span>
    of windows in this trip
  </p>
</div>
{% endif %}
</body>
</html>
"""

@app.route("/")
def index():
    trips = list_trips()
    selected = request.args.get("trip", trips[0])
    plot = None
    actual = None
    agreement = 0

    if selected:
        times, preds, actual_behavior = predict_trip(selected)
        plot = make_plot(times, preds, actual_behavior)
        actual = actual_behavior
        actual_idx = LABEL_NAMES.index(actual_behavior)
        agreement = sum(1 for p in preds if p == actual_idx) / len(preds)

    return render_template_string(PAGE, trips=trips, selected=selected, plot=plot, actual=actual, agreement=agreement)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)