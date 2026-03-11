# Smart Alzheimer Patient Monitoring Device

> An embedded AI-powered safety and health monitoring system for Alzheimer's patients — combining biometric identity verification, real-time health tracking, and GPS-based safety alerts on a Raspberry Pi 4B.

---

## GitHub Description (Short)

```
AI-powered Raspberry Pi device for Alzheimer's patients featuring face + voice biometric recognition, real-time heart rate & SpO2 monitoring via MAX30102, GPS safety alerts when patient wanders beyond 10m, and a live Tkinter dashboard — built with OpenCV, scikit-learn, and Python.
```

---

## Problem Statement

Alzheimer's patients face two critical daily risks:
- **Wandering** — they leave safe zones without caregivers knowing
- **Unidentified health emergencies** — unable to communicate their identity or symptoms

This device solves both by automatically identifying patients and continuously monitoring their health and location.

---

## Features

| Feature | Description |
|--------|-------------|
| **Face Recognition** | LBPH-based face recognition using OpenCV — no TensorFlow needed on Pi |
| **Voice Recognition** | SVM classifier trained on MFCC + spectral audio features |
| **Heart Rate Monitor** | Real-time BPM measurement via MAX30102 pulse oximeter |
| **SpO2 Monitor** | Blood oxygen percentage displayed live on OLED |
| **GPS Tracking** | NEO-M8N GPS module tracks patient location continuously |
| **Safety Alert** | On-screen warning when patient moves beyond 10 meters from home |
| **Live Dashboard** | Tkinter GUI showing camera feed, vitals, GPS, and recognition controls |
| **OLED Display** | 1.3" I2C display shows heart rate and SpO2 independently |

---

## Hardware Requirements

| Component | Purpose |
|-----------|---------|
| Raspberry Pi 4B (4GB RAM) | Main processing unit |
| USB Webcam | Face detection and recognition |
| USB Microphone | Voice recording and recognition |
| MAX30102 Sensor | Heart rate + SpO2 measurement |
| 1.3" OLED Display (I2C) | Health data display (SH1106 driver) |
| NEO-M8N GPS Module | Real-time location tracking |
| Breadboard + Jumper Wires | Sensor connections |

---

## Wiring Guide

### MAX30102 + OLED → Raspberry Pi (I2C)

```
MAX30102 / OLED          Raspberry Pi
─────────────────────────────────────
VCC          ──────────► Pin 1  (3.3V)
GND          ──────────► Pin 6  (GND)
SDA          ──────────► Pin 3  (GPIO2)
SCL          ──────────► Pin 5  (GPIO3)
```

> Both sensors share the same I2C pins via breadboard.
> MAX30102 → `0x57` | OLED → `0x3C`

### NEO-M8N GPS → Raspberry Pi (UART)

```
NEO-M8N                  Raspberry Pi
─────────────────────────────────────
VCC          ──────────► Pin 2  (5V)
GND          ──────────► Pin 14 (GND)
TX           ──────────► Pin 10 (RXD)
RX           ──────────► Pin 8  (TXD)
```

---

## Project Structure

```
face_voice_project/
├── dataset/
│   ├── faces/
│   │   ├── Person1/        # 10 photos per person
│   │   ├── Person2/
│   │   └── ...
│   └── voices/
│       ├── Person1/        # 10 .wav clips per person
│       └── ...
│
├── train_voice.py          # Train SVM voice model (run on PC)
├── export_for_pi_v2.py     # Export LBPH face model for Pi (run on PC)
│
├── recognize_pi.py         # Face + Voice recognition (Pi)
├── health_monitor.py       # MAX30102 + OLED health monitor (Pi)
├── dashboard.py            # Full GUI dashboard (Pi)
│
├── lbph_face_model.yml     # Trained face model
├── lbph_label_names.pkl    # Person name labels
├── voice_model.pkl         # Trained voice SVM model
│
├── test_oled.py            # Test OLED display
├── test_sensor.py          # Test MAX30102 sensor
└── test_gps.py             # Test GPS module
```

---

## Setup Guide

### Step 1 — Train Models on Windows PC

```bash
# Install dependencies
pip install deepface tf-keras opencv-python opencv-contrib-python
pip install librosa scikit-learn sounddevice numpy

# Collect dataset (10 face photos + 10 voice clips per person)
# Place in dataset/faces/<name>/ and dataset/voices/<name>/

# Train voice model
python train_voice.py

# Export face model for Pi
python export_for_pi_v2.py
```

### Step 2 — Setup Raspberry Pi

```bash
# Install system packages
sudo apt install -y python3-pip python3-venv portaudio19-dev \
  libgl1 libsm6 libxext6 libjpeg-dev libopenblas-dev \
  libgtk-3-dev i2c-tools python3-tk

# Create virtual environment
mkdir ~/face_voice_project && cd ~/face_voice_project
python3 -m venv env && source env/bin/activate

# Install Python packages
pip install numpy scikit-learn sounddevice librosa pillow
pip install opencv-python opencv-contrib-python
pip install smbus2 luma.oled pyserial pynmea2 scipy RPi.GPIO
```

### Step 3 — Enable Interfaces on Pi

```bash
sudo raspi-config
# Enable: I2C, Serial Port (no login shell, yes hardware)

# Add to /boot/firmware/config.txt under [all]:
# enable_uart=1
# dtoverlay=miniuart-bt

sudo reboot
```

### Step 4 — Transfer Files from PC to Pi

```bash
# Run in PowerShell on Windows PC
scp lbph_face_model.yml pi@<PI_IP>:/home/pi/face_voice_project/
scp lbph_label_names.pkl pi@<PI_IP>:/home/pi/face_voice_project/
scp voice_model.pkl pi@<PI_IP>:/home/pi/face_voice_project/
scp recognize_pi.py health_monitor.py dashboard.py pi@<PI_IP>:/home/pi/face_voice_project/
scp -r dataset pi@<PI_IP>:/home/pi/face_voice_project/
```

### Step 5 — Verify Sensors

```bash
# Check I2C sensors (should show 0x3C and 0x57)
i2cdetect -y 1

# Check GPS
sudo cat /dev/serial0

# Check camera
ls /dev/video*

# Check microphone
arecord -l
```

### Step 6 — Run the Project

```bash
cd ~/face_voice_project
source env/bin/activate

# Option A — Full dashboard (all features)
python3 dashboard.py

# Option B — Run separately
python3 recognize_pi.py    # Terminal 1: Face + Voice
python3 health_monitor.py  # Terminal 2: Health + OLED
```

---

## How It Works

```
Press START
     │
     ▼
┌─────────────┐
│  Face Scan  │  ← 10 seconds, look at camera
│  (LBPH)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Voice Scan  │  ← 5 seconds, speak clearly
│  (SVM)      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│   RESULT: CONFIRMED / NOT   │
│   Face name == Voice name?  │
└─────────────────────────────┘

Simultaneously running in background:
  ├── MAX30102 → Heart Rate + SpO2 → OLED Display
  └── NEO-M8N GPS → Distance from home → Alert if > 10m
```

---

## AI Model Details

### Face Recognition
- Algorithm: LBPH (Local Binary Pattern Histogram)
- Library: OpenCV `cv2.face.LBPHFaceRecognizer`
- Training: Haar Cascade detection + LBPH training on PC
- Inference: < 1 second on Pi 4B
- Confidence threshold: < 100 = recognized

### Voice Recognition
- Algorithm: SVM with RBF kernel (C=10, gamma='scale')
- Features: 86 per clip — MFCC (40), pitch, RMS energy, spectral centroid/rolloff
- Library: scikit-learn + librosa
- Accuracy: 100% on training set
- Audio: Stereo 16kHz recording, converted to mono

### Health Monitoring
- Heart Rate: FFT frequency analysis on IR signal (bandpass 0.5–4 Hz)
- SpO2: AC/DC ratio of Red and IR light (empirical formula)
- Smoothing: Rolling median over last 5 readings

---

## Performance on Raspberry Pi 4B

| Task | Time |
|------|------|
| Face recognition | ~1 second |
| Voice recognition | ~5 seconds (recording) + ~1s processing |
| Health reading | ~3 seconds per reading |
| GPS fix (first time) | 1–2 minutes |
| GPS fix (subsequent) | < 10 seconds |
| Dashboard refresh rate | 500ms |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera black screen | Use `cv2.CAP_V4L2`, warm up with 10 dummy frames |
| Face shows Unknown | Use `export_for_pi_v2.py` to create LBPH model |
| Mic channels error | Set `channels=2`, convert stereo→mono with `np.mean(audio, axis=1)` |
| GPS no data | Add `dtoverlay=miniuart-bt` to config.txt, reboot |
| WiFi stopped after GPS fix | Replace `dtoverlay=disable-bt` with `dtoverlay=miniuart-bt` |
| OLED blank | Use `sh1106` driver (not `ssd1306`) for 1.3" display |
| `cv2.face` error | Run `pip install opencv-contrib-python` |

---

## Dependencies

```txt
numpy
scikit-learn
sounddevice
librosa
pillow
opencv-python
opencv-contrib-python
smbus2
luma.oled
pyserial
pynmea2
scipy
RPi.GPIO
```

---

##  Target Users

- **Caregivers** — monitor patient identity, health, and location remotely via VNC
- **Hospitals & Care Homes** — low-cost embedded patient tracking solution
- **Families** — peace of mind with automatic safety alerts

---

## License

MIT License — free to use, modify, and distribute.

---

## Author

Built as an assistive technology project for Alzheimer's patient safety and health monitoring.

> *"Technology in service of dignity — helping patients stay safe and caregivers stay informed."*
