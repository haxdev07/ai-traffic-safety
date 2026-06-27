import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data"
IMG_SIZE = 96
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(torch.load("models/drowsiness_cnn.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

all_probs_fatigue = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)[:, 1]  # probability of "fatigue" class
        all_probs_fatigue.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

print("Classes:", val_ds.classes)

# Try several thresholds, lower = more sensitive to fatigue
for threshold in [0.5, 0.4, 0.35, 0.3, 0.25, 0.2]:
    preds = [1 if p >= threshold else 0 for p in all_probs_fatigue]
    cm = confusion_matrix(all_labels, preds)
    fn = cm[1][0]
    fp = cm[0][1]
    fatigue_recall = cm[1][1] / (cm[1][1] + cm[1][0])
    print(f"\nThreshold={threshold} | Fatigue Recall: {fatigue_recall:.4f} | FN: {fn} | FP: {fp}")