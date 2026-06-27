import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

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
ALL_DRIVERS = ["D1", "D2", "D3", "D4", "D5", "D6"]

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

def train_one_fold(test_driver, epochs=25):
    train_drivers = [d for d in ALL_DRIVERS if d != test_driver]

    train_windows, train_labels = load_trips(train_drivers)
    test_windows, test_labels = load_trips([test_driver])

    mean = train_windows.mean(axis=(0, 1))
    std = train_windows.std(axis=(0, 1))

    train_ds = DrivingDataset(train_windows, train_labels, mean, std)
    test_ds = DrivingDataset(test_windows, test_labels, mean, std)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    class_counts = np.array([max(Counter(train_labels)[i], 1) for i in range(3)])
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(DEVICE)

    model = LSTMClassifier(input_dim=len(FEATURE_COLS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return acc, all_labels, all_preds, len(train_labels), len(test_labels)


print("Running leave-one-driver-out cross-validation across all 6 drivers...\n")

results = []
all_labels_combined = []
all_preds_combined = []

for test_driver in ALL_DRIVERS:
    print(f"--- Fold: holding out {test_driver} ---")
    acc, labels, preds, n_train, n_test = train_one_fold(test_driver)
    print(f"  Train windows: {n_train}, Test windows: {n_test}, Accuracy: {acc:.4f}\n")
    results.append((test_driver, acc, n_test))
    all_labels_combined.extend(labels)
    all_preds_combined.extend(preds)

print("=" * 50)
print("LEAVE-ONE-DRIVER-OUT RESULTS")
print("=" * 50)
for driver, acc, n_test in results:
    print(f"  Held-out {driver}: accuracy = {acc:.4f}  (n={n_test})")

accs = [r[1] for r in results]
print(f"\nMean accuracy across folds: {np.mean(accs):.4f}")
print(f"Std deviation across folds: {np.std(accs):.4f}")
print(f"Min: {np.min(accs):.4f}  Max: {np.max(accs):.4f}")

print("\nCombined confusion matrix (all folds pooled):")
print(LABEL_NAMES)
print(confusion_matrix(all_labels_combined, all_preds_combined))
print("\nCombined classification report (all folds pooled):")
print(classification_report(all_labels_combined, all_preds_combined, target_names=LABEL_NAMES, digits=4))