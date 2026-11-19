# 🐱 Cat Hunter

An RC car that recognises cat breeds in real time with deep learning. Built for NUS module SWS3009A.

The car streams video from a Raspberry Pi. A 5-class breed classifier (Persian, Ragdoll,
Sphynx, Pallas, Singapura) labels the frames. The model runs either on a laptop (the
compute node) or on the Pi itself.

## Final model

| | |
| :--- | :--- |
| **Model** | ResNet50 (ImageNet-pretrained, fine-tuned on the 5-breed dataset) |
| **Accuracy** | **97.3%** on the 15% validation split |
| **Classes** | Persian, Ragdoll, Sphynx, Pallas, Singapura |

The ResNet50 model runs on the laptop compute node (see [Compute node](#compute-node)).

## System Architecture

```
                              Raspberry Pi                         Arduino
Browser (any device)          ┌──────────────────────────┐  USB    ┌──────────────┐
┌──────────────────┐   WiFi   │  web_control.py          │ Serial  │  Motor Shield│
│  Web Control UI  │◄────────►│  Flask + SocketIO        │◄───────►│  4 DC Motors │
│  (index.html)    │  WS/HTTP │                          │         │  Ultrasonic  │
│  Live camera     │◄─────────│  GET  /video_feed (MJPEG)│         │  Encoders    │
│  Breed + hunt    │          │  POST /api/detection     │         └──────────────┘
└──────────────────┘          └───────▲──────────┬───────┘
                                      │          │ MJPEG stream
                        detection JSON│          ▼
                              ┌──────────────────────────┐
                              │  Laptop: compute_node.py │
                              │  cat_detector.py         │
                              │  ResNet50 inference      │
                              └──────────────────────────┘
```

## Directory Structure

```
Cat_Hunter/
├── hardware_control/            # Runs on the Raspberry Pi (and the laptop compute node)
│   ├── web_control.py           # Flask + SocketIO server: camera stream, RC control, detection API
│   ├── car_controller.py        # Serial motor / ultrasonic sensor control for the Arduino
│   ├── cat_detector.py          # Inference wrapper around the trained model
│   ├── compute_node.py          # Laptop side: pulls the Pi stream, runs the model, posts results back
│   ├── camera_test.py           # Camera colour diagnostic for the Pi
│   └── templates/index.html     # Browser control UI
├── training_src/                # Model training pipeline (PyTorch)
│   ├── preprocess.py            # Stratified 85/15 train/val split -> metadata/*.csv
│   ├── dataset.py               # PyTorch Dataset + per-image augmentation
│   ├── baseline_model.py        # 3-layer CNN trained from scratch
│   ├── transfer_model.py        # ResNet50 / ConvNeXt-Tiny / MobileNetV2 transfer models
│   ├── train.py                 # Training loop (WeightedRandomSampler, MixUp/CutMix, cosine LR)
│   ├── evaluate.py              # Classification report + confusion matrix
│   └── inference_video.py       # Live breed prediction on a video stream or webcam
├── data_prep/                   # Building the dataset
│   ├── 1_balance_dataset.py     # Download the 5-breed dataset, downsample large classes to 400
│   ├── 2_scrape_missing.py      # Fill small classes up to 400 with Bing image search
│   └── extra/                   # Alternative scrapers (DuckDuckGo + Bing, 1000-image target)
├── deployment_tools/            # Training on a rented GPU server over SSH
│   ├── deploy_to_gpu.py         # Zip code + data/raw and upload it
│   └── run_remote.py            # Start the two-phase training run in the background
├── prototype/                   # Early 2-class (cat vs. non-cat) prototype and its results
├── models/                      # Trained weights (git-ignored)
├── .env.example                 # Template for secrets (copy to .env, never commit .env)
├── requirements.txt
├── LICENSE                      # MIT License
└── README.md
```

All commands below are run from the repository root unless stated otherwise.

## Data

The dataset has 5 classes: Persian, Ragdoll, Sphynx, Pallas and Singapura. Images go in
`data/raw/<Breed>/`, which is git-ignored.

1. **Balance the classes** ([`data_prep/1_balance_dataset.py`](data_prep/1_balance_dataset.py)).
   This script clones the team's 5-category dataset. Any class with more than 400 images
   is randomly downsampled to 400 (seed 42).
2. **Scrape missing images** ([`data_prep/2_scrape_missing.py`](data_prep/2_scrape_missing.py)).
   Classes with fewer than 400 images are filled up with Bing image search results
   (`"<breed> cat"`). The scripts in [`data_prep/extra/`](data_prep/extra/) are
   alternatives with a 1,000-image target:
   - `advanced_downloader.py` combines DuckDuckGo and Bing and runs several queries for the
     rarer breeds (Sphynx, Pallas, Singapura).
   - `balance_dataset.py` downsamples or upsamples every class to 1,000 with Bing.
   - `download_pallas.py` fetches Pallas cat images.
3. **85/15 stratified split** ([`training_src/preprocess.py`](training_src/preprocess.py)).
   Each class is shuffled (seed 42) and split 85% train / 15% validation, so both splits
   keep the class proportions. The output is `metadata/train_split.csv` and
   `metadata/val_split.csv`. There is no separate test split.
4. **Augmentation**:
   - Per image ([`training_src/dataset.py`](training_src/dataset.py)): resize to 224×224,
     random affine (±15°, 10% shift, 0.9–1.1 scale), horizontal flip, colour jitter
     (brightness, contrast and saturation 0.3), ImageNet normalisation.
   - Per batch ([`training_src/train.py`](training_src/train.py)): MixUp (α = 0.2) or
     CutMix (α = 1.0), chosen at random. `--no_aug` turns this off.
5. **WeightedRandomSampler**. The training loader samples each image with probability
   inversely proportional to its class size, so on average every breed is seen equally often.

## Hardware Requirements

- Raspberry Pi 4 (or 3B+)
- Arduino Uno/Mega + Motor Shield
- 4× DC motors (4WD chassis)
- USB camera or Pi Camera Module
- HC-SR04 ultrasonic sensor
- USB cable (Pi ↔ Arduino)

`car_controller.py` sends text commands over serial. The Arduino sketch that handles them
is not part of this repository.

## Software Setup

### 1. Clone and install

```bash
git clone https://github.com/dunja-g/Cat_Hunter.git
cd Cat_Hunter
pip install -r requirements.txt
```

On Raspberry Pi OS, also install `picamera2` if you use the Pi Camera Module.

### 2. Configure secrets (optional)

```bash
cp .env.example .env
```

`.env` is git-ignored. It holds:
- the GPU server login for `deployment_tools/` (`GPU_HOST`, `GPU_PORT`, `GPU_USER`,
  `GPU_PASSWORD`, `GPU_REMOTE_DIR`)
- `FLASK_SECRET_KEY` for `web_control.py`. If it is not set, a random key is generated at
  every start.

### 3. Build the dataset

```bash
python data_prep/1_balance_dataset.py
python data_prep/2_scrape_missing.py
```

### 4. Train and evaluate

```bash
# Stratified 85/15 split -> metadata/train_split.csv, metadata/val_split.csv
python training_src/preprocess.py

# Phase 1: frozen ResNet50 backbone, train the new classification head
python training_src/train.py --model transfer --arch resnet50 --epochs 10 --no_aug

# Phase 2: unfreeze the backbone and fine-tune from the phase-1 weights
python training_src/train.py --model transfer --arch resnet50 --epochs 30 --fine_tune --weights models/best_transfer.pth

# Classification report + confusion matrix on the validation split
python training_src/evaluate.py --model transfer
```

The best checkpoint (lowest validation loss) is saved to `models/best_transfer.pth`.
Curves and confusion matrices go to `plots/`. Other options:
- `--model baseline` trains the CNN from scratch.
- `--arch convnext_tiny` or `--arch mobilenet_v2` picks another backbone.

To run the same two phases on a rented GPU server, fill in `.env`, then run:

```bash
python deployment_tools/deploy_to_gpu.py   # upload code + data/raw
python deployment_tools/run_remote.py      # start training in the background (log: /root/train.log)
```

### 5. Run the car

**Offloaded mode (ResNet50 on a laptop, recommended).** On the Pi:

```bash
python3 hardware_control/web_control.py /dev/ttyACM0 --no-ml
```

On a laptop on the same network:

```bash
python hardware_control/compute_node.py --pi-ip <raspberry-pi-ip> --weights models/best_transfer.pth
```

**Standalone mode (everything on the Pi).** In this mode, `web_control.py` builds a
MobileNetV2 and runs inference once per second. You need a MobileNetV2 checkpoint, trained
with `--arch mobilenet_v2`. The ResNet50 checkpoint does not load here.

```bash
python3 hardware_control/web_control.py /dev/ttyACM0 --weights models/best_transfer.pth
```

### 6. Open the control interface

Go to `http://<raspberry-pi-ip>:5000` in any browser.

## Compute node

To keep the Pi free for the camera and motors, inference can run on a laptop instead:

1. The Pi runs `web_control.py --no-ml`. It serves the camera as an MJPEG stream at
   `/video_feed` and accepts detection results at `POST /api/detection`.
2. The laptop runs [`compute_node.py`](hardware_control/compute_node.py), which pulls the
   stream. A background thread reads frames continuously and keeps only the latest one, so
   OpenCV's buffer never builds up lag. The script reconnects after 10 failed reads in a row.
3. Every `--interval` seconds (default 0.5 s), it runs the model on the latest frame through
   `CatDetector` (`--model resnet50`, confidence threshold `--threshold`, default 0.65).
4. It POSTs `{detected, confidence, label, inference_ms}` as JSON to the Pi's
   `/api/detection`.
5. The Pi handles that result exactly like a local detection. It pushes the result to the
   browser over SocketIO and drives Hunt Mode.

## Features

- **Remote Control**: drive the car from any browser over WebSocket
- **Live Camera**: MJPEG video stream (camera set to 15 FPS)
- **Breed Detection**: 5-class classifier, on the laptop compute node or on the Pi
- **Hunt Mode**: the car drives forward while a cat is detected and stops when it loses it
- **Obstacle Avoidance**: the ultrasonic sensor triggers an emergency stop (can be toggled)
- **Keyboard Controls**: WASD + Q/E for curves, arrow keys, space to stop
- **4-Speed Gears**: PWM 130 / 170 / 210 / 255
- **Mobile-Friendly**: touch controls with a responsive dark-themed UI

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

## Prototype

[`prototype/`](prototype/) contains the earlier 2-class (cat vs. non-cat) prototype: a
baseline CNN and a ResNet transfer model, with their plots and results. Those results are
for the 2-class task, not for the final 5-breed model above.

## License

MIT License. See [LICENSE](LICENSE) for details.
