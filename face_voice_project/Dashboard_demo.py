# dashboard_demo.py - LAPTOP DEMO VERSION
# Hardware (MAX30102, GPS, OLED) is simulated
# Face recognition, Voice recognition, Camera are REAL

import sys
import platform
import threading
import time
import math
import random
import tkinter as tk
from tkinter import ttk

# ═══════════════════════════════════════
#   MOCK HARDWARE-ONLY MODULES
# ═══════════════════════════════════════
from unittest.mock import MagicMock
sys.modules['smbus2']                     = MagicMock()
sys.modules['serial']                     = MagicMock()
sys.modules['pynmea2']                    = MagicMock()
sys.modules['luma']                       = MagicMock()
sys.modules['luma.core']                  = MagicMock()
sys.modules['luma.core.interface']        = MagicMock()
sys.modules['luma.core.interface.serial'] = MagicMock()
sys.modules['luma.core.render']           = MagicMock()
sys.modules['luma.oled']                  = MagicMock()
sys.modules['luma.oled.device']           = MagicMock()

# ── Now safe to import everything ──
import cv2
import numpy as np
import pickle
import sounddevice as sd
import librosa
from PIL import Image, ImageTk

OLED_OK = False
GPS_OK  = False

# ═══════════════════════════════════════
#           LOAD AI MODELS
# ═══════════════════════════════════════
print("Loading models...")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load face recognizer
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("lbph_face_model.yml")
    with open("lbph_label_names.pkl", "rb") as f:
        label_names = pickle.load(f)
    FACE_MODEL_OK = True
    print("✓ Face model loaded")
except Exception as e:
    FACE_MODEL_OK = False
    label_names   = {}
    recognizer    = None
    print(f"✗ Face model not found: {e}")

# Load voice recognizer
try:
    with open("voice_model.pkl", "rb") as f:
        voice_data = pickle.load(f)
    voice_clf = voice_data["model"]
    voice_le  = voice_data["encoder"]
    VOICE_MODEL_OK = True
    print("✓ Voice model loaded")
except Exception as e:
    VOICE_MODEL_OK = False
    voice_clf      = None
    voice_le       = None
    print(f"✗ Voice model not found: {e}")

print("Models ready!\n")

# ═══════════════════════════════════════
#           SHARED STATE
# ═══════════════════════════════════════
state = {
    # Health (simulated)
    "hr":           0,
    "spo2":         0,
    "finger_on":    False,

    # GPS (simulated)
    "lat":          28.613939,
    "lon":          77.209023,
    "satellites":   8,
    "home_lat":     28.613939,
    "home_lon":     77.209023,
    "distance":     0.0,
    "gps_fix":      True,
    "alert":        False,

    # Recognition
    "face_name":    "---",
    "voice_name":   "---",
    "confirmed":    False,
    "scanning":     False,

    # Camera
    "frame":        None,
}

hr_history   = []
spo2_history = []

# ═══════════════════════════════════════
#           HELPER FUNCTIONS
# ═══════════════════════════════════════
def haversine(lat1, lon1, lat2, lon2):
    R    = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = (math.sin(dphi/2)**2 +
            math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def smooth(val, history, max_len=5):
    if val > 0:
        history.append(val)
        if len(history) > max_len:
            history.pop(0)
    return int(np.median(history)) if history else 0

# ═══════════════════════════════════════
#     SIMULATED HEALTH SENSOR THREAD
# ═══════════════════════════════════════
def health_thread():
    """Simulate MAX30102 readings for demo"""
    print("→ Health thread started (SIMULATED)")
    base_hr   = 72
    base_spo2 = 98
    step      = 0
    while True:
        # Simulate finger detection toggling for realism
        # In demo it's always "on" after 2 seconds
        if step < 2:
            state["finger_on"] = False
        else:
            state["finger_on"] = True
            # Slight random variation ±3 BPM, ±1 SpO2
            hr_raw   = base_hr   + random.randint(-3, 3)
            spo2_raw = base_spo2 + random.randint(-1, 1)
            spo2_raw = max(95, min(100, spo2_raw))
            state["hr"]   = smooth(hr_raw,   hr_history)
            state["spo2"] = smooth(spo2_raw, spo2_history)
        step += 1
        time.sleep(1)

# ═══════════════════════════════════════
#       SIMULATED GPS THREAD
# ═══════════════════════════════════════
def gps_thread():
    """Simulate slow GPS drift for demo"""
    print("→ GPS thread started (SIMULATED)")
    drift = 0.0
    while True:
        # Tiny random drift each cycle
        drift += random.uniform(-0.000005, 0.000005)
        state["lat"] = state["home_lat"] + drift
        state["lon"] = state["home_lon"] + drift * 0.5

        if state["home_lat"] is not None:
            dist = haversine(
                state["home_lat"], state["home_lon"],
                state["lat"],      state["lon"]
            )
            state["distance"] = dist
            state["alert"]    = dist > 10

        state["satellites"] = random.randint(7, 12)
        time.sleep(2)

# ═══════════════════════════════════════
#           CAMERA THREAD
# ═══════════════════════════════════════
def camera_thread():
    """Capture real laptop webcam frames"""
    print("→ Camera thread started (REAL)")
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam.isOpened():
        print("✗ Could not open webcam!")
        return

    # Flush buffer
    for _ in range(5):
        cam.read()

    while True:
        ret, frame = cam.read()
        if ret:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 1.1, 5, minSize=(50, 50))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y),
                              (x+w, y+h), (0, 200, 255), 2)
                cv2.putText(frame, "Face Detected",
                            (x, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 200, 255), 1)
            state["frame"] = frame
        time.sleep(0.03)

# ═══════════════════════════════════════
#       FACE + VOICE RECOGNITION
# ═══════════════════════════════════════
def do_face_recognition(frame):
    if not FACE_MODEL_OK:
        return "Model not loaded", 0
    try:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return "No face", 0
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        roi         = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        label, conf = recognizer.predict(roi)
        if conf < 100:
            return label_names[label], conf
        return "Unknown", conf
    except Exception as e:
        print("Face error:", e)
        return "Error", 0

def do_voice_recognition():
    if not VOICE_MODEL_OK:
        return "Model not loaded", 0
    try:
        print("Recording 5s audio...")
        audio = sd.rec(int(5 * 16000), samplerate=16000,
                       channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        try:
            audio, _ = librosa.effects.trim(audio, top_db=20)
        except:
            pass

        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
        mm   = np.mean(mfcc.T, axis=0)
        ms   = np.std(mfcc.T,  axis=0)
        p, _ = librosa.piptrack(y=audio, sr=16000)
        pm   = np.mean(p[p > 0]) if np.any(p > 0) else 0
        ps   = np.std(p[p > 0])  if np.any(p > 0) else 0
        rms  = librosa.feature.rms(y=audio)
        sc   = np.mean(librosa.feature.spectral_centroid(y=audio, sr=16000))
        sr2  = np.mean(librosa.feature.spectral_rolloff(y=audio,  sr=16000))
        feat = np.concatenate(
            [mm, ms, [pm, ps, np.mean(rms), np.std(rms), sc, sr2]]
        ).reshape(1, -1)

        prob = voice_clf.predict_proba(feat)[0]
        idx  = np.argmax(prob)
        return voice_le.inverse_transform([idx])[0], prob[idx]

    except Exception as e:
        print("Voice error:", e)
        return "Error", 0

# ═══════════════════════════════════════
#              MAIN GUI
# ═══════════════════════════════════════
class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Health & Security Dashboard  [DEMO MODE]")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1024x620")

        self.font_title = ("Arial", 11, "bold")
        self.font_value = ("Arial", 20, "bold")
        self.font_small = ("Arial", 9)
        self.font_label = ("Arial", 10)
        self.font_alert = ("Arial", 14, "bold")

        self.build_gui()
        self.update_gui()

    def card(self, parent, title, row, col,
             rowspan=1, colspan=1, color="#16213e"):
        frame = tk.Frame(parent, bg=color, relief="flat", bd=0)
        frame.grid(row=row, column=col,
                   rowspan=rowspan, columnspan=colspan,
                   padx=5, pady=5, sticky="nsew")
        tk.Label(frame, text=title,
                 bg=color, fg="#00d4ff",
                 font=self.font_title).pack(anchor="w", padx=8, pady=(6, 2))
        tk.Frame(frame, bg="#00d4ff", height=1).pack(fill="x", padx=8)
        return frame

    def build_gui(self):
        # ── Demo mode banner ──
        demo_bar = tk.Label(
            self.root,
            text="⚙  DEMO MODE  —  Hardware sensors simulated  |  "
                 "Camera · Face · Voice are LIVE",
            bg="#0f3460", fg="#00d4ff",
            font=("Arial", 9, "bold")
        )
        demo_bar.pack(fill="x")

        # ── Alert Banner ──
        self.alert_banner = tk.Label(
            self.root,
            text="⚠  ALERT: Person has moved beyond 10 meters!",
            bg="#ff0000", fg="white",
            font=self.font_alert
        )

        # ── Main grid ──
        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(fill="both", expand=True, padx=10, pady=6)

        for i in range(3):
            main.columnconfigure(i, weight=1)
        for i in range(3):
            main.rowconfigure(i, weight=1)

        # ── CARD 1: Camera Feed ──
        cam_card = self.card(main, "📷 Live Camera", 0, 0, rowspan=2)
        self.cam_label = tk.Label(cam_card, bg="#0a0a1a",
                                  text="Starting camera...",
                                  fg="#555566")
        self.cam_label.pack(fill="both", expand=True, padx=8, pady=8)

        # ── CARD 2: Heart Rate ──
        hr_card = self.card(main, "❤  Heart Rate  (simulated)", 0, 1)
        self.hr_value = tk.Label(
            hr_card, text="---",
            bg="#16213e", fg="#ff4757",
            font=self.font_value)
        self.hr_value.pack(pady=5)
        tk.Label(hr_card, text="BPM",
                 bg="#16213e", fg="#ffffff",
                 font=self.font_label).pack()
        self.finger_status = tk.Label(
            hr_card, text="Initializing...",
            bg="#16213e", fg="#ffa502",
            font=self.font_small)
        self.finger_status.pack(pady=3)

        # ── CARD 3: SpO2 ──
        spo2_card = self.card(main, "🩸 SpO2  (simulated)", 0, 2)
        self.spo2_value = tk.Label(
            spo2_card, text="---",
            bg="#16213e", fg="#2ed573",
            font=self.font_value)
        self.spo2_value.pack(pady=5)
        tk.Label(spo2_card, text="%",
                 bg="#16213e", fg="#ffffff",
                 font=self.font_label).pack()
        self.spo2_status = tk.Label(
            spo2_card, text="Initializing...",
            bg="#16213e", fg="#ffa502",
            font=self.font_small)
        self.spo2_status.pack(pady=3)

        # ── CARD 4: GPS ──
        gps_card = self.card(main, "📍 GPS Location  (simulated)", 1, 1)
        self.gps_lat  = tk.Label(gps_card, text="Lat: ---",
                                 bg="#16213e", fg="#ffffff",
                                 font=self.font_label)
        self.gps_lat.pack(anchor="w", padx=10, pady=2)
        self.gps_lon  = tk.Label(gps_card, text="Lon: ---",
                                 bg="#16213e", fg="#ffffff",
                                 font=self.font_label)
        self.gps_lon.pack(anchor="w", padx=10, pady=2)
        self.gps_dist = tk.Label(gps_card, text="Distance: 0.0 m",
                                 bg="#16213e", fg="#00d4ff",
                                 font=self.font_label)
        self.gps_dist.pack(anchor="w", padx=10, pady=2)
        self.gps_sats = tk.Label(gps_card, text="Satellites: ---",
                                 bg="#16213e", fg="#ffffff",
                                 font=self.font_small)
        self.gps_sats.pack(anchor="w", padx=10, pady=2)
        tk.Button(gps_card, text="Set Home Location",
                  bg="#0f3460", fg="white",
                  font=self.font_small,
                  command=self.set_home).pack(pady=5)

        # ── CARD 5: Recognition ──
        rec_card = self.card(main, "👤 Face & Voice Recognition  (LIVE)", 1, 2)
        self.face_label = tk.Label(
            rec_card, text="Face:  ---",
            bg="#16213e", fg="#ffffff",
            font=self.font_label)
        self.face_label.pack(anchor="w", padx=10, pady=3)
        self.voice_label = tk.Label(
            rec_card, text="Voice: ---",
            bg="#16213e", fg="#ffffff",
            font=self.font_label)
        self.voice_label.pack(anchor="w", padx=10, pady=3)
        self.result_label = tk.Label(
            rec_card, text="Press FACE SCAN to begin",
            bg="#16213e", fg="#ffa502",
            font=("Arial", 10, "bold"))
        self.result_label.pack(pady=3)

        btn_frame = tk.Frame(rec_card, bg="#16213e")
        btn_frame.pack(fill="x", padx=8, pady=5)
        self.face_btn = tk.Button(
            btn_frame, text="FACE SCAN",
            bg="#0f3460", fg="white",
            font=self.font_small,
            command=self.start_face_scan)
        self.face_btn.pack(side="left", padx=3, expand=True, fill="x")
        self.voice_btn = tk.Button(
            btn_frame, text="VOICE SCAN",
            bg="#0f3460", fg="white",
            font=self.font_small,
            command=self.start_voice_scan,
            state="disabled")
        self.voice_btn.pack(side="left", padx=3, expand=True, fill="x")

        # ── CARD 6: Status bar ──
        status_card = self.card(
            main, "System Status", 2, 0, colspan=3, color="#0f3460")
        self.status_label = tk.Label(
            status_card,
            text="Demo mode ready  —  Face model: " +
                 ("✓" if FACE_MODEL_OK else "✗ not found") +
                 "   Voice model: " +
                 ("✓" if VOICE_MODEL_OK else "✗ not found"),
            bg="#0f3460", fg="#00d4ff",
            font=self.font_small)
        self.status_label.pack(pady=5)

    # ── Button handlers ──────────────────────────────────────
    def set_home(self):
        state["home_lat"] = state["lat"]
        state["home_lon"] = state["lon"]
        state["distance"] = 0.0
        state["alert"]    = False
        self.status_label.config(
            text="Home location updated to current simulated position!")

    def start_face_scan(self):
        if state["scanning"]:
            return
        state["scanning"] = True
        self.face_btn.config(state="disabled", text="Scanning...")
        self.voice_btn.config(state="disabled")
        self.result_label.config(
            text="Face scanning for 10s — look at camera...",
            fg="#ffa502")
        threading.Thread(target=self._face_scan_thread, daemon=True).start()

    def _face_scan_thread(self):
        for i in range(10, 0, -1):
            self.status_label.config(
                text=f"Face scanning... {i}s remaining — keep still, look at camera")
            time.sleep(1)

        if state["frame"] is not None:
            name, score = do_face_recognition(state["frame"])
            state["face_name"] = name
            color = "#2ed573" if name not in \
                    ["Unknown", "No face", "Error", "Model not loaded"] \
                    else "#ff4757"
            self.face_label.config(
                text=f"Face:  {name}  (conf: {round(score, 1)})",
                fg=color)
            self.result_label.config(
                text="Face scan done! Click VOICE SCAN next",
                fg="#ffa502")
            self.voice_btn.config(state="normal")
        else:
            self.result_label.config(
                text="No camera frame — check webcam",
                fg="#ff4757")

        self.face_btn.config(state="normal", text="FACE SCAN")
        state["scanning"] = False

    def start_voice_scan(self):
        if state["scanning"]:
            return
        state["scanning"] = True
        self.voice_btn.config(state="disabled", text="Listening...")
        self.result_label.config(
            text="🎤 Speak now — recording 5 seconds...",
            fg="#ffa502")
        threading.Thread(target=self._voice_scan_thread, daemon=True).start()

    def _voice_scan_thread(self):
        for i in range(5, 0, -1):
            self.status_label.config(
                text=f"🎤 Recording voice... {i}s remaining — speak clearly")
            time.sleep(1)

        name, conf = do_voice_recognition()
        state["voice_name"] = name

        confirmed = (
            state["face_name"] == name and
            name not in ["---", "Unknown", "No face",
                         "Error", "Model not loaded"]
        )
        state["confirmed"] = confirmed

        self.voice_label.config(
            text=f"Voice: {name}  ({round(conf*100)}%)",
            fg="#2ed573" if confirmed else "#ff4757")

        if confirmed:
            self.result_label.config(
                text=f"✓  CONFIRMED: {name}", fg="#2ed573")
        else:
            self.result_label.config(
                text=f"✗  NOT CONFIRMED  "
                     f"(Face: {state['face_name']} | Voice: {name})",
                fg="#ff4757")

        self.voice_btn.config(state="normal", text="VOICE SCAN")
        self.status_label.config(text="Recognition complete!")
        state["scanning"] = False

    # ── GUI update loop ──────────────────────────────────────
    def update_gui(self):
        # Alert banner
        if state["alert"]:
            self.alert_banner.pack(fill="x")
        else:
            self.alert_banner.pack_forget()

        # Camera
        if state["frame"] is not None:
            frame = cv2.cvtColor(state["frame"], cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(frame)
            img   = img.resize((280, 210))
            photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=photo)
            self.cam_label.image = photo

        # Heart Rate
        hr = state["hr"]
        self.hr_value.config(text=str(hr) if hr > 0 else "---")
        if state["finger_on"]:
            self.finger_status.config(
                text="Finger detected (simulated)", fg="#2ed573")
        else:
            self.finger_status.config(
                text="Initializing simulation...", fg="#ffa502")

        # SpO2
        sp = state["spo2"]
        self.spo2_value.config(text=str(sp) if sp > 0 else "---")
        if sp >= 95:
            self.spo2_status.config(text="Normal", fg="#2ed573")
        elif sp >= 90:
            self.spo2_status.config(text="Low", fg="#ffa502")
        elif sp > 0:
            self.spo2_status.config(text="Very Low!", fg="#ff4757")
        else:
            self.spo2_status.config(text="Initializing...", fg="#ffa502")

        # GPS
        self.gps_lat.config(
            text=f"Lat: {round(state['lat'], 6)}")
        self.gps_lon.config(
            text=f"Lon: {round(state['lon'], 6)}")
        dist = state["distance"]
        self.gps_dist.config(
            text=f"Distance: {round(dist, 1)} m",
            fg="#ff4757" if state["alert"] else "#00d4ff")
        self.gps_sats.config(
            text=f"Satellites: {state['satellites']}")

        self.root.after(500, self.update_gui)


# ═══════════════════════════════════════
#              START APP
# ═══════════════════════════════════════
if __name__ == "__main__":
    threads = [
        threading.Thread(target=health_thread, daemon=True),
        threading.Thread(target=gps_thread,    daemon=True),
        threading.Thread(target=camera_thread, daemon=True),
    ]
    for t in threads:
        t.start()

    print("Starting dashboard (demo mode)...")
    root = tk.Tk()
    app  = Dashboard(root)
    root.mainloop()