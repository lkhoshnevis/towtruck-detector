#!/usr/bin/env python3
"""
Truck detector — YOLOv8 + Raspberry Pi Camera (ribbon/CSI)
- Detects trucks (COCO class 7)
- Sends Telegram photo alert with 30s cooldown
- Logs FPS/CPU/RAM/network to CSV every 60s
- Headless (no display window)
"""

import cv2
import requests
import time
import csv
import psutil
import os
import io
import logging
import threading
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from dotenv import load_dotenv

load_dotenv()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
from picamera2 import Picamera2
from ultralytics import YOLO, settings
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
CHAT_ID          = os.environ["CHAT_ID"]
TRUCK_CLASS      = 7       # COCO class 7 = truck
COOLDOWN         = 30      # seconds between Telegram alerts
LOG_INTERVAL     = 60      # seconds between CSV rows
LOG_FILE         = os.path.expanduser("~/truck_stats.csv")
MODEL_NAME       = "yolov8n_ncnn_model"
CONF_THRESH      = 0.45
SHOW_PREVIEW     = False  # set True when HDMI is connected locally
STREAM_PORT      = 9090   # MJPEG stream port — view at http://localhost:9090 via SSH tunnel
CAPTURE_SIZE     = (2592, 1944) # full sensor resolution, no binning = sharpest
TARGET_FPS       = 15           # lock camera + loop to this rate to eliminate buffer lag
TEMP_WARN_C      = 80           # °C — send Telegram alert above this

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.expanduser("~/truck_detector.log")),
    ],
)
log = logging.getLogger(__name__)

def read_temp() -> float:
    try:
        raw = open("/sys/class/thermal/thermal_zone0/temp").read()
        return int(raw.strip()) / 1000.0
    except Exception:
        return 0.0

# Suppress YOLO telemetry
settings.update({"sync": False})

# ── Telegram ──────────────────────────────────────────────────────────────────
def telegram_message(text: str, retries: int = 3, delay: float = 5.0):
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                data={"chat_id": CHAT_ID, "text": text},
                timeout=10,
            )
            data = r.json()
            if data.get("ok"):
                log.info(f"Telegram message sent (attempt {attempt})")
                return
            else:
                log.error(f"Telegram message rejected (attempt {attempt}): {data}")
        except Exception as e:
            log.error(f"Telegram message error (attempt {attempt}): {e}")
        if attempt < retries:
            time.sleep(delay)
    log.error("Telegram message failed after all retries")

def telegram_photo(frame, caption: str):
    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            log.error("JPEG encoding failed")
            return
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
            data={"chat_id": CHAT_ID, "caption": caption},
            files={"photo": ("truck.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
            timeout=20,
        )
        log.info("Telegram photo sent.")
    except Exception as e:
        log.error(f"Telegram photo error: {e}")

# ── CSV ───────────────────────────────────────────────────────────────────────
def init_csv():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "uptime_min", "fps",
                "cpu_pct", "ram_used_mb", "ram_total_mb",
                "net_sent_mb", "net_recv_mb",
                "trucks_this_period",
            ])
        log.info(f"CSV created: {LOG_FILE}")

def write_csv_row(uptime_min, fps, period_trucks, net_baseline):
    mem  = psutil.virtual_memory()
    cpu  = psutil.cpu_percent(interval=None)
    net  = psutil.net_io_counters()
    sent = (net.bytes_sent - net_baseline.bytes_sent) / 1e6
    recv = (net.bytes_recv - net_baseline.bytes_recv) / 1e6
    row  = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"{uptime_min:.1f}",
        f"{fps:.2f}",
        f"{cpu:.1f}",
        f"{mem.used / 1e6:.1f}",
        f"{mem.total / 1e6:.1f}",
        f"{sent:.3f}",
        f"{recv:.3f}",
        period_trucks,
    ]
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)
    log.info(
        f"Stats | {uptime_min:.1f}min | FPS:{fps:.1f} | CPU:{cpu:.1f}% | "
        f"RAM:{mem.used/1e6:.0f}/{mem.total/1e6:.0f}MB | "
        f"Net↑{sent:.2f}MB ↓{recv:.2f}MB | Trucks:{period_trucks}"
    )

# ── MJPEG stream ─────────────────────────────────────────────────────────────
# Use a dict so threads can mutate it without needing `global`
frame_buf  = {"jpg": b""}
frame_lock = threading.Lock()
run_event  = threading.Event()   # set = detection active
stop_event = threading.Event()   # set = detection should pause

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with frame_lock:
                    jpg = frame_buf["jpg"]
                if jpg:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
                time.sleep(1 / TARGET_FPS)
        except Exception:
            pass
    def log_message(self, *a):
        pass

def poll_telegram_commands():
    """Long-poll for incoming messages; reply to 'any opps?' with a live snapshot."""
    offset = None
    while True:
        try:
            params = {"timeout": 10, "allowed_updates": ["message"]}
            if offset is not None:
                params["offset"] = offset
            r = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                params=params,
                timeout=15,
            )
            data = r.json()
            if not data.get("ok"):
                time.sleep(5)
                continue
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                msg  = update.get("message", {})
                text = msg.get("text", "").strip().lower()
                cid  = msg.get("chat", {}).get("id")
                if not cid:
                    continue
                if text == "a":
                    with frame_lock:
                        jpg = bytes(frame_buf["jpg"])
                    if jpg:
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                            data={"chat_id": cid, "caption": "👀 Here's what I see right now"},
                            files={"photo": ("snap.jpg", io.BytesIO(jpg), "image/jpeg")},
                            timeout=20,
                        )
                        log.info(f"Snapshot sent in reply to 'a' from {cid}")
                    else:
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                            data={"chat_id": cid, "text": "Camera not ready yet, try again."},
                            timeout=10,
                        )
                elif text == "r":
                    if run_event.is_set():
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                            data={"chat_id": cid, "text": "Already running."},
                            timeout=10,
                        )
                    else:
                        log.info("Start command received via Telegram.")
                        stop_event.clear()
                        run_event.set()
                elif text == "s":
                    if not run_event.is_set():
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                            data={"chat_id": cid, "text": "Not running."},
                            timeout=10,
                        )
                    else:
                        log.info("Stop command received via Telegram.")
                        stop_event.set()
                elif text == "status":
                    mem  = psutil.virtual_memory()
                    cpu  = psutil.cpu_percent(interval=0.5)
                    temp = read_temp()
                    freq = int(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read()) // 1000
                    disk = psutil.disk_usage("/")
                    state = "🟢 Running" if run_event.is_set() else "🔴 Stopped"
                    msg = (
                        f"📊 Pi Status\n"
                        f"Detector: {state}\n"
                        f"Temp: {temp:.1f}°C\n"
                        f"CPU: {cpu:.1f}% @ {freq} MHz\n"
                        f"RAM: {mem.used/1e6:.0f} / {mem.total/1e6:.0f} MB\n"
                        f"Disk: {disk.used/1e9:.1f} / {disk.total/1e9:.1f} GB"
                    )
                    requests.post(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                        data={"chat_id": cid, "text": msg},
                        timeout=10,
                    )
                    log.info(f"Status sent to {cid}")
        except Exception as e:
            log.error(f"Telegram poll error: {e}")
            time.sleep(5)

def start_stream_server():
    try:
        server = ThreadedHTTPServer(("0.0.0.0", STREAM_PORT), MJPEGHandler)
        log.info(f"MJPEG stream on port {STREAM_PORT}")
        server.serve_forever()
    except OSError as e:
        log.error(f"MJPEG stream server failed to start: {e}")

# ── Detection loop ────────────────────────────────────────────────────────────
def run_detection(cam, model):
    start_time       = time.time()
    last_notify      = 0.0
    last_log         = time.time()
    net_baseline     = psutil.net_io_counters()
    fps_frame_count  = 0
    fps_window_start = time.time()
    current_fps      = 0.0
    period_trucks    = 0

    frame_interval = 1.0 / TARGET_FPS
    log.info("Detection loop running.")
    while not stop_event.is_set():
        loop_start = time.time()
        try:
            frame = cam.capture_array()
        except Exception as e:
            log.error(f"Camera capture failed: {e}")
            telegram_message(f"📷 Camera error — detection stopped: {e}\nCheck the ribbon cable.")
            break

        # ── crop: keep 60% of frame (zoom in 30%), shifted up 50px ──────
        h, w = frame.shape[:2]
        crop_h = int(h * 0.60)
        crop_w = int(w * 0.60)
        y1 = max(0, (h - crop_h) // 2 - 50)
        x1 = (w - crop_w) // 2
        frame = frame[y1:y1 + crop_h, x1:x1 + crop_w]

        # ── Inference ─────────────────────────────────────────────────────
        results = model(frame, classes=[TRUCK_CLASS],
                        conf=CONF_THRESH, imgsz=320, verbose=False)
        boxes   = results[0].boxes
        now     = time.time()

        # ── Annotate every frame for preview ──────────────────────────────
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 80, 255), 2)
            cv2.putText(display, f"Truck {conf:.0%}", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
        cv2.putText(display, f"FPS: {current_fps:.1f}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ── Push to MJPEG stream ───────────────────────────────────────────
        _, jpg_buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            frame_buf["jpg"] = jpg_buf.tobytes()

        if SHOW_PREVIEW:
            cv2.imshow("Truck Detector", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if len(boxes) > 0:
            period_trucks += len(boxes)
            if now - last_notify >= COOLDOWN:
                last_notify = now
                n = len(boxes)
                caption = (
                    f"🚛 {n} truck{'s' if n > 1 else ''} detected!\n"
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"FPS: {current_fps:.1f}"
                )
                log.info(caption.replace("\n", " | "))
                threading.Thread(
                    target=telegram_photo,
                    args=(display.copy(), caption),
                    daemon=True,
                ).start()
            else:
                remaining = int(COOLDOWN - (now - last_notify))
                log.info(f"Truck detected (cooldown: {remaining}s remaining)")

        # ── FPS throttle ──────────────────────────────────────────────────
        elapsed = time.time() - loop_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

        # ── FPS ───────────────────────────────────────────────────────────
        fps_frame_count += 1
        elapsed_win = now - fps_window_start
        if elapsed_win >= 5.0:
            current_fps      = fps_frame_count / elapsed_win
            fps_frame_count  = 0
            fps_window_start = now

        # ── CSV logging + temp check ───────────────────────────────────────
        if now - last_log >= LOG_INTERVAL:
            uptime_min = (now - start_time) / 60
            write_csv_row(uptime_min, current_fps, period_trucks, net_baseline)
            period_trucks = 0
            last_log = now
            temp = read_temp()
            log.info(f"CPU temp: {temp:.1f}°C")
            if temp >= TEMP_WARN_C:
                telegram_message(f"🌡️ Warning: Pi temperature is {temp:.1f}°C — consider cooling!")

    run_event.clear()
    log.info("Detection loop stopped.")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_csv()

    log.info(f"Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    log.info("Model ready.")

    log.info("Opening Pi Camera (CSI)…")
    cam = Picamera2()
    frame_us = int(1_000_000 / TARGET_FPS)
    config = cam.create_preview_configuration(
        main={"size": CAPTURE_SIZE, "format": "RGB888"},
        controls={
            "FrameDurationLimits": (frame_us, frame_us),
            "AeEnable": True,   # auto-exposure on
            "AwbEnable": True,  # auto white balance on
        },
        buffer_count=2,
    )
    cam.configure(config)
    cam.start()
    time.sleep(1)
    log.info("Camera open.")

    threading.Thread(target=start_stream_server, daemon=True).start()
    threading.Thread(target=poll_telegram_commands, daemon=True).start()

    run_event.set()  # start detecting immediately

    try:
        while True:
            run_event.wait()          # block until "run" (or startup)
            stop_event.clear()
            telegram_message("🟢 Truck detector online and watching.")
            run_detection(cam, model)
            telegram_message("🔴 Truck detector stopped.")
    except KeyboardInterrupt:
        stop_event.set()
        log.info("Stopped by user.")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        log.info(f"Done. Stats saved to {LOG_FILE}")

if __name__ == "__main__":
    main()
