import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image

DATA_ROOT = "data/CrashBest"
CSV_PATH = "data/Crash_Table.csv"
OUTPUT_DIR = "data/crash_features"
IMG_SIZE = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

backbone = mobilenet_v3_small(weights="IMAGENET1K_V1")
backbone.classifier = nn.Identity()
backbone = backbone.to(DEVICE)
backbone.eval()

def get_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = backbone(x)
    return feat.squeeze(0).cpu().numpy()

videos = []
with open(CSV_PATH, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        vidname = row[0]
        frame_labels = [int(x) for x in row[1:51]]
        videos.append((vidname, frame_labels))

print(f"Total videos in CSV: {len(videos)}")

processed = 0
skipped = 0

for vidname, frame_labels in videos:
    vid_id = vidname.zfill(6)
    out_path = os.path.join(OUTPUT_DIR, f"{vid_id}.npz")
    if os.path.exists(out_path):
        continue

    features = []
    missing = False
    for frame_num in range(1, 51):
        img_path = os.path.join(DATA_ROOT, f"C_{vid_id}_{frame_num:02d}.jpg")
        if not os.path.exists(img_path):
            missing = True
            break
        feat = get_feature(img_path)
        features.append(feat)

    if missing:
        skipped += 1
        continue

    features = np.array(features, dtype=np.float32)
    labels = np.array(frame_labels, dtype=np.int64)
    np.savez(out_path, features=features, labels=labels)

    processed += 1
    if processed % 50 == 0:
        print(f"Processed {processed} videos so far... (skipped {skipped})")

print(f"\nDone. Processed: {processed}, Skipped (missing frames): {skipped}")
