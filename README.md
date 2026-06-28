# AI Traffic Safety System

Real machine learning rebuild of a rule-based AI dashcam safety pitch — three trained models, each properly evaluated, each with a live demo.

| Phase                 | What it does                                | Model               | Result                              |
| --------------------- | ------------------------------------------- | ------------------- | ----------------------------------- |
| 1. Drowsiness         | Detects driver fatigue from webcam          | MobileNetV3 CNN     | 89.1% val acc, 95.2% fatigue recall |
| 2. Driving Behavior   | Classifies normal/drowsy/aggressive driving | LSTM on sensor data | 86.6% mean cross-driver accuracy    |
| 3. Accident Detection | Detects crashes from dashcam video          | CNN features + LSTM | 89.9% test acc, 90.7% crash recall  |

See `PROJECT.md` for full details and how to run each demo.

Built with AI assistance (Claude) for setup, debugging, and implementation — choices, testing, and direction were mine.
