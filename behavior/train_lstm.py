import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

DATA_ROOT = "data/UAH-DRIVESET-v1"
WINDOW_SIZE = 30
STEP_SIZE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
LABEL_MAP = {"NORMAL": 0, "DROWSY": 1, "AGGRESSIVE": 2}
LABEL_NAMES = ["NORMAL", "DROWSY", "AGGRESSIVE"]

TRAIN_DRIVERS = ["D1", "D2", "D3", "D4"]
VAL_DRIVERS = ["D5"]
TEST_DRIVERS = ["D6"]

def parse_trip_folder_name(folder_name):
    parts = folder_name.split("-")
    behavior = re.sub(r"\d+$", "", parts[3])
    return behavior

def load_trips(driver_list):
    windows = []
    labels = []
    for driver_folder in sorted(os.listdir(DATA_ROOT)):
        if driver_folder not in driver_list:
            continue
        driver_path = os.path.join(DATA_ROOT, driver_folder)
        if not os.path.isdir(driver_path):
            continue
        for trip_folder in sorted(os.listdir(driver_path)):
            trip_path = os.path.join(driver_path, trip_folder)
            semantic_file = os.path.join(trip_path, "SEMANTIC_ONLINE.txt")
            if not os.path.exists(semantic_file):
                continue
            behavior = parse_trip_folder_name(trip_folder)
            if behavior not in LABEL_MAP:
                continue
            label = LABEL_MAP[behavior]
            df = pd.read_csv(semantic_file, sep=r"\s+", header=None, names=COLUMNS)
            features = df[FEATURE_COLS].values.astype(np.float32)
            for start in range(0, len(features) - WINDOW_SIZE, STEP_SIZE):
                window = features[start:start + WINDOW_SIZE]
                windows.append(window)
                labels.append(label)
    return np.array(windows), np.array(labels)

class DrivingDataset(Dataset):
    def __init__(self, windows, labels, mean, std):
        self.windows = (windows - mean) / (std + 1e-6)
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)

print("Loading training data...")
train_windows, train_labels = load_trips(TRAIN_DRIVERS)
print("Loading validation data...")
val_windows, val_labels = load_trips(VAL_DRIVERS)
print("Loading test data...")
test_windows, test_labels = load_trips(TEST_DRIVERS)

print(f"\nTrain windows: {len(train_labels)}, distribution: {Counter(train_labels)}")
print(f"Val windows: {len(val_labels)}, distribution: {Counter(val_labels)}")
print(f"Test windows: {len(test_labels)}, distribution: {Counter(test_labels)}")

mean = train_windows.mean(axis=(0, 1))
std = train_windows.std(axis=(0, 1))

train_ds = DrivingDataset(train_windows, train_labels, mean, std)
val_ds = DrivingDataset(val_windows, val_labels, mean, std)
test_ds = DrivingDataset(test_windows, test_labels, mean, std)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

class_counts = np.array([Counter(train_labels)[i] for i in range(3)])
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(DEVICE)

model = LSTMClassifier(input_dim=len(FEATURE_COLS)).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 30
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_correct, train_total, train_loss = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        train_correct += (out.argmax(1) == y).sum().item()
        train_total += x.size(0)

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += x.size(0)

    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/behavior_lstm.pth")
        np.save("models/behavior_norm_mean.npy", mean)
        np.save("models/behavior_norm_std.npy", std)
        print(f"  -> Saved new best model (val_acc={val_acc:.4f})")

print(f"\nTraining done. Best val accuracy: {best_val_acc:.4f}")

model.load_state_dict(torch.load("models/behavior_lstm.pth"))
model.eval()
test_correct, test_total = 0, 0
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        preds = out.argmax(1)
        test_correct += (preds == y).sum().item()
        test_total += x.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

print(f"\nTest accuracy on held-out driver D6: {test_correct/test_total:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
print("\nConfusion Matrix (rows=actual, cols=predicted):")
print(LABEL_NAMES)
print(confusion_matrix(all_labels, all_preds))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES, digits=4))