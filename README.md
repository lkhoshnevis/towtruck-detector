# Towtruck Detector

A lightweight, always-on truck detection system running on a Raspberry Pi. Captures live video from a CSI camera, runs YOLOv8 inference locally, and sends instant Telegram alerts with a photo when a truck is spotted — no cloud required.

---

## Hardware

| Component | Recommended |
|---|---|
| **Raspberry Pi** | Raspberry Pi 5 (4GB or 8GB) |
| **Camera** | Raspberry Pi Camera Module 3 (CSI ribbon cable) |
| **Power** | Official 27W USB-C PD power supply |
| **Storage** | 32GB+ microSD (Class 10 / A2) |
| **Cooling** | Active cooler recommended for sustained inference |

The camera connects via the CSI ribbon cable (not USB). The Pi Camera Module 3 is preferred for its autofocus and full-sensor 12MP resolution — the script captures at 2592×1944 and crops to the center 60% for a natural zoom.

---

## How It Works

```
Pi Camera (CSI)
      │
      ▼
Picamera2 capture @ 2592×1944, 15 FPS
      │
      ▼
Center-crop to 60% of frame (zoom effect, shifted up 50px)
      │
      ▼
YOLOv8n (NCNN, runs on-device) — detects COCO class 7 (truck)
      │
      ├── Truck found ──► Annotate frame ──► Telegram photo alert (30s cooldown)
      │
      ├── Always ──────► MJPEG stream on port 9090 (view via SSH tunnel)
      │
      └── Every 60s ───► Log FPS / CPU / RAM / temp / network to CSV
```

- **Model**: YOLOv8n converted to NCNN format — optimized for ARM, runs fully on-device with no GPU
- **Confidence threshold**: 45% — adjust `CONF_THRESH` in the script to tune sensitivity
- **Alert cooldown**: 30 seconds between Telegram notifications to avoid spam
- **Temperature guard**: Sends a warning if the Pi exceeds 80°C

---

## Telegram Commands

Once the bot is running, send these commands from your Telegram chat:

| Command | Action |
|---|---|
| `a` | Send a live snapshot of what the camera sees right now |
| `r` | Resume detection (if stopped) |
| `s` | Stop detection |
| `status` | Show Pi stats: temp, CPU, RAM, disk, detector state |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/lkhoshnevis/towtruck-detector.git
cd towtruck-detector
```

### 2. Install dependencies

```bash
pip install ultralytics picamera2 opencv-python-headless requests psutil python-dotenv
```

### 3. Create your `.env` file

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```
TELEGRAM_TOKEN=your_bot_token_here
CHAT_ID=your_chat_id_here
```

- Create a bot via [@BotFather](https://t.me/BotFather) to get your token
- Get your chat ID by messaging [@userinfobot](https://t.me/userinfobot)

### 4. Download the NCNN model

Place the `yolov8n_ncnn_model/` folder in the same directory as the script. You can export it from the standard YOLOv8n weights:

```bash
yolo export model=yolov8n.pt format=ncnn
```

### 5. Run manually

```bash
python3 truck_detector_pi.py
```

### 6. Run as a system service (auto-start on boot)

```bash
sudo cp truck-detector.service /etc/systemd/system/
sudo systemctl enable truck-detector
sudo systemctl start truck-detector
```

Check status:

```bash
sudo systemctl status truck-detector
journalctl -u truck-detector -f
```

---

## Live Stream

While running, an MJPEG stream is available on port 9090. Forward it over SSH to view on your laptop:

```bash
ssh -L 9090:localhost:9090 pi@<pi-ip>
```

Then open `http://localhost:9090` in your browser.

---

## Photos

### Camera Setup
![Camera mounted on Pi](images/camera_setup.jpg)

### Detection in Action
![Truck detected with bounding box](images/detection_result.jpg)

### Telegram Alert
![Telegram notification with annotated photo](images/telegram_alert.jpg)

> Replace these placeholder images with your own. Put photos in an `images/` folder in the repo root.

---

## Who Can Use This

Anyone with a Raspberry Pi 5 and a CSI camera who wants passive, local truck detection — no subscription, no cloud, no fees. Useful for:

- Monitoring a driveway or street for delivery trucks / tow trucks
- Security setups where you want truck-specific alerts
- Learning how to run YOLOv8 on edge hardware

---

## License

MIT
