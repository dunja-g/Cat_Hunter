# 🐱 Cat Hunter

An autonomous RC car that detects cats in real time using deep learning. Built for NUS module SWS3009A.

## System Architecture

```
Browser (any device)          Raspberry Pi                    Arduino
┌─────────────────┐    WiFi   ┌──────────────────────┐  USB   ┌──────────────┐
│  Web Control UI  │◄────────►│  Flask + SocketIO    │◄──────►│  Motor Shield│
│  (index.html)    │  WS/HTTP │  web_control.py      │ Serial │  4 DC Motors │
│                  │          │                      │        │  Ultrasonic  │
│  Live Camera     │◄─────────│  MJPEG stream        │        │  Encoders    │
│  Cat Detection   │◄─────────│  cat_detector.py     │        └──────────────┘
│  Drive Controls  │          │  MobileNetV2 (1 FPS) │
└─────────────────┘          └──────────────────────┘
```

## Directory Structure

```
Cat_Hunter/
├── car_controller.py       # Low-level Arduino serial motor/sensor control
├── web_control.py          # Flask + SocketIO web server (camera, RC, detection)
├── cat_detector.py         # Real-time cat classifier (MobileNetV2 inference)
├── templates/
│   └── index.html          # Browser-based RC control UI
├── cat_detection/          # Deep learning model training pipeline
│   ├── src/
│   │   ├── baseline_model.py     # 3-layer CNN from scratch
│   │   ├── transfer_model.py     # MobileNetV2 / ResNet50 transfer learning
│   │   ├── dataset.py            # PyTorch Dataset + augmentation
│   │   ├── preprocess.py         # Image scanning + stratified CSV splits
│   │   ├── train.py              # Training loop + checkpointing
│   │   └── evaluate.py           # Test evaluation + confusion matrix
│   ├── plots/                    # Training curves & confusion matrices
│   ├── cat_transfer_learning_project.py   # All-in-one notebook script
│   ├── kaggle_cat_not_cat_complete.py     # Polished Kaggle version
│   ├── tasks.md                  # Development task checklist
│   └── README.md                 # ML pipeline documentation (Chinese)
├── models/                 # Trained model weights (git-ignored)
├── requirements.txt
├── LICENSE                 # MIT License
└── README.md               # This file
```

## Features

- **Remote Control** — Drive the car from any browser via WebSocket (zero latency)
- **Live Camera** — MJPEG video stream at ~15 FPS
- **Cat Detection** — MobileNetV2 classifies camera frames at ~1 FPS
- **Hunt Mode** — Car automatically drives toward detected cats
- **Obstacle Avoidance** — Ultrasonic sensor stops the car before collisions
- **Keyboard Controls** — WASD + Q/E for curves, arrow keys, space to stop
- **4-Speed Gears** — PWM 130 / 170 / 210 / 255
- **Mobile-Friendly** — Touch controls with responsive dark-themed UI

## Hardware Requirements

- Raspberry Pi 4 (or 3B+)
- Arduino Uno/Mega + Motor Shield
- 4× DC motors (4WD chassis)
- USB camera or Pi Camera Module
- HC-SR04 ultrasonic sensor
- USB cable (Pi ↔ Arduino)

## Software Setup

### 1. Clone the repository

```bash
git clone https://github.com/dunja-g/Cat_Hunter.git
cd Cat_Hunter
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (or use pretrained weights)

```bash
cd cat_detection

# Prepare dataset splits
python src/preprocess.py

# Train transfer learning model
python src/train.py --model transfer --epochs 10

# Evaluate
python src/evaluate.py --model transfer
```

The trained model will be saved to `models/best_transfer.pth`.

### 4. Run the server (on Raspberry Pi)

```bash
python web_control.py /dev/ttyACM0 models/best_transfer.pth
```

### 5. Open the control interface

Navigate to `http://<raspberry-pi-ip>:5000` in any browser.

## Keyboard Controls

| Key | Action |
|-----|--------|
| `W` / `↑` | Forward |
| `S` / `↓` | Backward |
| `A` / `←` | Turn Left |
| `D` / `→` | Turn Right |
| `Q` | Curve Left |
| `E` | Curve Right |
| `Z` | Back-Curve Left |
| `X` | Back-Curve Right |
| `Space` | Stop |
| `1-4` | Set Gear |

## Team

- **Person A** — Data preprocessing, baseline CNN, training loop
- **Person B** — Augmentation, transfer learning, hyperparameter tuning

## License

MIT License — see [LICENSE](LICENSE) for details.