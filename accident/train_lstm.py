import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

FEATURES_DIR = "data/crash_features"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_files = sorted(os.listdir(FEATURES_DIR))
print(f"Total feature files: {len(all_files)}")

np.random.seed(42)
indices = np.random.permutation(len(all_files))
n_train = int(0.8 * len(all_files))
n_val = int(0.1 * len(all_files))

train_files = [all_files[i] for i in indices[:n_train]]
val_files = [all_files[i] for i in indices[n_train:n_train+n_val]]
test_files = [all_files[i] for i in indices[n_train+n_val:]]

print(f"Train videos: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

class CrashDataset(Dataset):
    def __init__(self, file_list, mean=None, std=None):
        self.file_list = file_list
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(os.path.join(FEATURES_DIR, self.file_list[idx]))
        features = data["features"]
        labels = data["labels"]
        if self.mean is not None:
            features = (features - self.mean) / (self.std + 1e-6)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

sample_feats = []
for f in train_files[:200]:
    data = np.load(os.path.join(FEATURES_DIR, f))
    sample_feats.append(data["features"])
sample_feats = np.concatenate(sample_feats, axis=0)
mean = sample_feats.mean(axis=0)
std = sample_feats.std(axis=0)

train_ds = CrashDataset(train_files, mean, std)
val_ds = CrashDataset(val_files, mean, std)
test_ds = CrashDataset(test_files, mean, std)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

all_train_labels = []
for f in train_files:
    data = np.load(os.path.join(FEATURES_DIR, f))
    all_train_labels.extend(data["labels"].tolist())
label_counts = Counter(all_train_labels)
print(f"Frame-level label distribution in train: {dict(label_counts)}")

class AccidentLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits

sample = np.load(os.path.join(FEATURES_DIR, train_files[0]))
feature_dim = sample["features"].shape[1]
print(f"Feature dimension: {feature_dim}")

class_weights = torch.tensor(
    [1.0 / max(label_counts[0], 1), 1.0 / max(label_counts[1], 1)],
    dtype=torch.float32
)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(DEVICE)

model = AccidentLSTM(input_dim=feature_dim).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_correct, train_total, train_loss = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, 2), y.reshape(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        preds = logits.argmax(-1)
        train_correct += (preds == y).sum().item()
        train_total += y.numel()

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(-1)
            val_correct += (preds == y).sum().item()
            val_total += y.numel()

    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_files):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/accident_lstm.pth")
        np.save("models/accident_norm_mean.npy", mean)
        np.save("models/accident_norm_std.npy", std)
        print(f"  -> Saved new best model (val_acc={val_acc:.4f})")

print(f"\nTraining done. Best val accuracy: {best_val_acc:.4f}")

model.load_state_dict(torch.load("models/accident_lstm.pth"))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(-1)
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(y.cpu().numpy().flatten())

print(f"\nTest accuracy (frame-level): {(np.array(all_preds) == np.array(all_labels)).mean():.4f}")
print("\nConfusion Matrix (rows=actual, cols=predicted):")
print(["NO_CRASH", "CRASH"])
print(confusion_matrix(all_labels, all_preds))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["NO_CRASH", "CRASH"], digits=4))