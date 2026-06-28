# AI Traffic Safety System — Project Status

Real ML rebuild of a rule-based AI dashcam pitch deck. All 3 phases complete.

## Phase 1 — Drowsiness Detection ✅ DONE

- MediaPipe face crop + MobileNetV3 CNN, trained on UTA-RLDD (9k images)
- 89.1% val accuracy, tuned threshold (0.25) gives 95.2% fatigue recall
- Live webcam demo via Flask (`app.py`, port 5000)
- Fixed real train/test mismatch bug (raw webcam frame vs face-crop training data)

## Phase 2 — Driving Behavior Classification ✅ DONE

- LSTM on UAH-DriveSet sensor-score windows (6 drivers, 40 trips)
- Leave-one-driver-out cross-validation: 86.6% mean accuracy (range 68-96%)
- Final deployable model trained on all 6 drivers
- Live demo via Flask (`behavior_app.py`, port 5001) — pick any trip, see predicted timeline

## Phase 3 — Accident Detection ✅ DONE

- MobileNetV3 frame features + LSTM, trained on CCD (1,500 crash videos, 75k frames)
- 89.9% test accuracy, tuned threshold (0.2) gives 90.7% crash recall
- Live demo via Flask (`accident_app.py`, port 5002) — pick any video, see frame-by-frame crash probability

## Running the demos

```cmd
venv311\Scripts\activate
python app.py              # Phase 1, localhost:5000
python behavior_app.py     # Phase 2, localhost:5001
python accident\accident_app.py   # Phase 3, localhost:5002
```

## Datasets used

- UTA-RLDD (drowsiness) — Kaggle mirror
- UAH-DriveSet (driving behavior) — robesafe.com
- CCD / Car Crash Dataset (accidents) — Kaggle mirror

## Tech stack

Python, PyTorch, OpenCV, MediaPipe, Flask, scikit-learn, pandas

## Cut from original scope (infra, not ML)

Tolling/fine automation, license plate OCR, biometric verification, P2P enforcement network, crowd-sourced traffic routing.
