import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
import time

DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transform)

print("Classes:", train_ds.classes)  # confirm 0=active, 1=fatigue order

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Class weights for imbalance (5862 active, 3192 fatigue)
class_counts = torch.tensor([5862.0, 3192.0])
class_weights = (1.0 / class_counts)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(DEVICE)

# Lightweight transfer learning model — feasible on CPU
model = mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += imgs.size(0)

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += imgs.size(0)

    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    elapsed = time.time() - start

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/train_total:.4f} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/drowsiness_cnn.pth")
        print(f"  -> Saved new best model (val_acc={val_acc:.4f})")

print(f"\nTraining done. Best val accuracy: {best_val_acc:.4f}")
print("Model saved to models/drowsiness_cnn.pth")