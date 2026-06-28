import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

FEATURES_DIR = "data/crash_features"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_files = sorted(os.listdir(FEATURES_DIR))
np.random.seed(42)
indices = np.random.permutation(len(all_files))
n_train = int(0.8 * len(all_files))
n_val = int(0.1 * len(all_files))
test_files = [all_files[i] for i in indices[n_train+n_val:]]

mean = np.load("models/accident_norm_mean.npy")
std = np.load("models/accident_norm_std.npy")

class CrashDataset(Dataset):
    def __init__(self, file_list, mean, std):
        self.file_list = file_list
        self.mean = mean
        self.std = std
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        data = np.load(os.path.join(FEATURES_DIR, self.file_list[idx]))
        features = (data["features"] - self.mean) / (self.std + 1e-6)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(data["labels"], dtype=torch.long)

class AccidentLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

test_ds = CrashDataset(test_files, mean, std)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

sample = np.load(os.path.join(FEATURES_DIR, test_files[0]))
feature_dim = sample["features"].shape[1]

model = AccidentLSTM(input_dim=feature_dim).to(DEVICE)
model.load_state_dict(torch.load("models/accident_lstm.pth", map_location=DEVICE))
model.eval()

all_probs_crash = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        probs = F.softmax(logits, dim=-1)[..., 1]
        all_probs_crash.extend(probs.cpu().numpy().flatten())
        all_labels.extend(y.numpy().flatten())

all_probs_crash = np.array(all_probs_crash)
all_labels = np.array(all_labels)

print("Threshold sweep (lower threshold = more sensitive to crash detection):\n")
for threshold in [0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]:
    preds = (all_probs_crash >= threshold).astype(int)
    cm = confusion_matrix(all_labels, preds)
    fn = cm[1][0]
    fp = cm[0][1]
    crash_recall = cm[1][1] / (cm[1][1] + cm[1][0])
    crash_precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
    print(f"Threshold={threshold} | Crash Recall: {crash_recall:.4f} | Crash Precision: {crash_precision:.4f} | FN: {fn} | FP: {fp}")