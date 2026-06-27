import torch
import torch.nn as nn
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

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("Classes:", val_ds.classes)  # ['active', 'fatigue'] -> 0=active, 1=fatigue
print("\nConfusion Matrix:")
print("         Pred Active  Pred Fatigue")
cm = confusion_matrix(all_labels, all_preds)
print(f"Active      {cm[0][0]:5d}        {cm[0][1]:5d}")
print(f"Fatigue     {cm[1][0]:5d}        {cm[1][1]:5d}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=val_ds.classes, digits=4))

fn = cm[1][0]  # actual fatigue, predicted active -- THE DANGEROUS ERROR
fp = cm[0][1]  # actual active, predicted fatigue -- annoying but safe
print(f"\nFALSE NEGATIVES (missed fatigue cases): {fn} -- this is the metric that matters most")
print(f"FALSE POSITIVES (false alarms): {fp}")