# dashboard.py - Complete Health + GPS + Face/Voice Dashboard
import tkinter as tk
from tkinter import ttk, font
import threading
import time
import cv2
import numpy as np
import pickle
import sounddevice as sd
import librosa
import smbus2
import serial
import pynmea2
from PIL import Image, ImageTk
from luma.core.interface.serial import i2c as i2c_serial
from luma.oled.device import sh1106
from luma.core.render import canvas
import math

# ═══════════════════════════════════════
#           LOAD AI MODELS
# ═══════════════════════════════════════
print("Loading models...")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("lbph_face_model.yml")
with open("lbph_label_names.pkl", "rb") as f:
    label_names = pickle.load(f)

with open("voice_model.pkl", "rb") as f:
    voice_data = pickle.load(f)
voice_clf = voice_data["model"]
voice_le  = voice_data["encoder"]

print("Models loaded!")

# ═══════════════════════════════════════
#           SENSOR SETUP
# ═══════════════════════════════════════

# MAX30102
bus       = smbus2.SMBus(1)
MAX_ADDR  = 0x57

def init_max30102():
    try:
        bus.write_byte_data(MAX_ADDR, 0x09, 0x40)
        time.sleep(0.1)
        bus.write_byte_data(MAX_ADDR, 0x09, 0x03)
        bus.write_byte_data(MAX_ADDR, 0x0A, 0x67)
        bus.write_byte_data(MAX_ADDR, 0x0C, 0x7F)
        bus.write_byte_data(MAX_ADDR, 0x0D, 0x7F)
        bus.write_byte_data(MAX_ADDR, 0x08, 0x4F)
        return True
    except:
        return False

# OLED
try:
    oled_serial = i2c_serial(port=1, address=0x3C)
    oled        = sh1106(oled_serial)
    OLED_OK     = True
except:
    OLED_OK = False
    print("OLED not found")

# GPS
try:
    gps_serial = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)
    GPS_OK     = True
except:
    GPS_OK = False
    print("GPS not found")

# ═══════════════════════════════════════
#           SHARED STATE
# ═══════════════════════════════════════
state = {
    # Health
    "hr":           0,
    "spo2":         0,
    "finger_on":    False,

    # GPS
    "lat":          0.0,
    "lon":          0.0,
    "satellites":   0,
    "home_lat":     None,
    "home_lon":     None,
    "distance":     0.0,
    "gps_fix":      False,
    "alert":        False,

    # Recognition
    "face_name":    "---",
    "voice_name":   "---",
    "confirmed":    False,
    "scanning":     False,
    "scan_status":  "Press START to begin",

    # Camera
    "frame":        None,
}

hr_history   = []
spo2_history = []

# ═══════════════════════════════════════
#           HELPER FUNCTIONS
# ═══════════════════════════════════════
def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two GPS coordinates"""
    R    = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = (math.sin(dphi/2)**2 +
            math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def smooth(val, history, max_len=5):
    if val > 0:
        history.append(val)
        if len(history) > max_len:
            history.pop(0)
    return int(np.median(history)) if history else 0

# ═══════════════════════════════════════
#           SENSOR THREADS
# ═══════════════════════════════════════
def health_thread():
    """Continuously read MAX30102"""
    init_max30102()
    while True:
        try:
            data     = bus.read_i2c_block_data(MAX_ADDR, 0x07, 6)
            ir_quick = (data[3] << 16 | data[4] << 8 | data[5]) & 0x3FFFF

            if ir_quick < 50000:
                state["finger_on"] = False
                hr_history.clear()
                spo2_history.clear()
                time.sleep(0.3)
                continue

            state["finger_on"] = True

            # Collect samples
            red_vals, ir_vals = [], []
            for _ in range(300):
                d   = bus.read_i2c_block_data(MAX_ADDR, 0x07, 6)
                red = (d[0] << 16 | d[1] << 8 | d[2]) & 0x3FFFF
                ir  = (d[3] << 16 | d[4] << 8 | d[5]) & 0x3FFFF
                red_vals.append(red)
                ir_vals.append(ir)
                time.sleep(0.01)

            red_arr = np.array(red_vals)
            ir_arr  = np.array(ir_vals)

            # HR via FFT
            try:
                from scipy.signal import butter, filtfilt
                ac   = ir_arr - np.mean(ir_arr)
                nyq  = 50
                b, a = butter(2, [0.5/nyq, 4.0/nyq], btype='band')
                filt = filtfilt(b, a, ac)
                fft  = np.abs(np.fft.rfft(filt))
                freq = np.fft.rfftfreq(len(filt), 1/100)
                rng  = (freq >= 0.5) & (freq <= 4.0)
                if np.any(rng):
                    pf  = freq[rng][np.argmax(fft[rng])]
                    hr_raw = int(pf * 60)
                    if not (40 <= hr_raw <= 200):
                        hr_raw = 0
                else:
                    hr_raw = 0
            except:
                hr_raw = 0

            # SpO2
            try:
                R    = (np.std(red_arr)/np.mean(red_arr)) / \
                       (np.std(ir_arr)/np.mean(ir_arr))
                sp   = int(104 - 17 * R)
                spo2_raw = max(80, min(100, sp))
            except:
                spo2_raw = 0

            state["hr"]   = smooth(hr_raw,   hr_history)
            state["spo2"] = smooth(spo2_raw, spo2_history)
            time.sleep(1)

        except Exception as e:
            time.sleep(1)

def gps_thread():
    """Continuously read GPS"""
    if not GPS_OK:
        return
    while True:
        try:
            line = gps_serial.readline().decode(
                'ascii', errors='replace').strip()
            if 'GGA' in line:
                msg = pynmea2.parse(line)
                if msg.latitude != 0:
                    state["lat"]        = msg.latitude
                    state["lon"]        = msg.longitude
                    state["satellites"] = int(msg.num_sats)
                    state["gps_fix"]    = True

                    # Set home on first fix
                    if state["home_lat"] is None:
                        state["home_lat"] = msg.latitude
                        state["home_lon"] = msg.longitude
                        print("Home location set!")

                    # Calculate distance from home
                    if state["home_lat"] is not None:
                        dist = haversine(
                            state["home_lat"], state["home_lon"],
                            msg.latitude,      msg.longitude
                        )
                        state["distance"] = dist
                        state["alert"]    = dist > 10  # 10 meter alert
        except:
            pass

def camera_thread():
    """Continuously capture camera frames"""
    cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(10):
        cam.read()
    while True:
        ret, frame = cam.read()
        if ret:
            # Draw face box
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 1.1, 5, minSize=(50, 50))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,200,255), 2)
            state["frame"] = frame
        time.sleep(0.03)

def oled_thread():
    """Update OLED with health data"""
    if not OLED_OK:
        return
    while True:
        try:
            with canvas(oled) as draw:
                draw.text((15, 0),  "HEALTH MONITOR", fill="white")
                draw.line([(0,12),(128,12)], fill="white", width=1)
                if not state["finger_on"]:
                    draw.text((5,  25), "Place finger",  fill="white")
                    draw.text((15, 42), "on sensor...",  fill="white")
                else:
                    hr_text = str(state["hr"])   + " BPM" \
                              if state["hr"] > 0   else "--- BPM"
                    sp_text = str(state["spo2"]) + " %"  \
                              if state["spo2"] > 0 else "--- %"
                    draw.text((0, 18), "Heart Rate:",    fill="white")
                    draw.text((0, 32), hr_text,          fill="white")
                    draw.line([(0,44),(128,44)],
                              fill="white", width=1)
                    draw.text((0,  48), "SpO2:",         fill="white")
                    draw.text((50, 48), sp_text,         fill="white")
        except:
            pass
        time.sleep(2)

# ═══════════════════════════════════════
#       FACE + VOICE RECOGNITION
# ═══════════════════════════════════════
def do_face_recognition(frame):
    try:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 1.1, 5, minSize=(60,60))
        if len(faces) == 0:
            return "No face", 0
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        roi         = cv2.resize(gray[y:y+h, x:x+w], (100,100))
        label, conf = recognizer.predict(roi)
        if conf < 100:
            return label_names[label], conf
        return "Unknown", conf
    except:
        return "No face", 0

def do_voice_recognition():
    try:
        audio = sd.rec(int(5*16000), samplerate=16000,
                       channels=2, dtype='float32', device=1)
        sd.wait()
        audio = np.mean(audio, axis=1)
        try:
            audio, _ = librosa.effects.trim(audio, top_db=20)
        except:
            pass
        mfcc  = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
        mm    = np.mean(mfcc.T, axis=0)
        ms    = np.std(mfcc.T,  axis=0)
        p, _  = librosa.piptrack(y=audio, sr=16000)
        pm    = np.mean(p[p>0]) if np.any(p>0) else 0
        ps    = np.std(p[p>0])  if np.any(p>0) else 0
        rms   = librosa.feature.rms(y=audio)
        sc    = np.mean(librosa.feature.spectral_centroid(y=audio, sr=16000))
        sr2   = np.mean(librosa.feature.spectral_rolloff(y=audio,  sr=16000))
        feat  = np.concatenate(
            [mm, ms, [pm, ps, np.mean(rms), np.std(rms), sc, sr2]]
        ).reshape(1,-1)
        prob  = voice_clf.predict_proba(feat)[0]
        idx   = np.argmax(prob)
        return voice_le.inverse_transform([idx])[0], prob[idx]
    except Exception as e:
        print("Voice error:", e)
        return "Error", 0

def recognition_sequence():
    """Run face then voice recognition"""
    state["scanning"]    = True
    state["confirmed"]   = False
    state["face_name"]   = "---"
    state["voice_name"]  = "---"

    # Face scan 10 seconds
    state["scan_status"] = "STEP 1: Face scanning (10s)..."
    time.sleep(10)
    if state["frame"] is not None:
        name, score = do_face_recognition(state["frame"])
        state["face_name"]   = name
        state["scan_status"] = "Face: " + name + " | Press SPACE for voice"

    # Wait for space (handled by GUI button)
    state["scanning"]    = False
    state["scan_status"] = "Face done! Click VOICE SCAN"

def voice_sequence():
    state["scanning"]    = True
    state["scan_status"] = "STEP 2: Speak now (5s)..."
    name, conf = do_voice_recognition()
    state["voice_name"]  = name
    state["confirmed"]   = (state["face_name"] == name and
                            name not in ["---","Unknown","No face","Error"])
    state["scan_status"] = ("CONFIRMED: " + name
                            if state["confirmed"] else "NOT CONFIRMED")
    state["scanning"]    = False

# ═══════════════════════════════════════
#              MAIN GUI
# ═══════════════════════════════════════
class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Health & Security Dashboard")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1024x600")

        # Fonts
        self.font_title  = ("Arial", 11, "bold")
        self.font_value  = ("Arial", 20, "bold")
        self.font_small  = ("Arial", 9)
        self.font_label  = ("Arial", 10)
        self.font_alert  = ("Arial", 14, "bold")

        self.build_gui()
        self.update_gui()

    def card(self, parent, title, row, col,
             rowspan=1, colspan=1, color="#16213e"):
        """Create a styled card"""
        frame = tk.Frame(parent, bg=color,
                         relief="flat", bd=0)
        frame.grid(row=row, column=col,
                   rowspan=rowspan, columnspan=colspan,
                   padx=5, pady=5, sticky="nsew")
        tk.Label(frame, text=title,
                 bg=color, fg="#00d4ff",
                 font=self.font_title).pack(
                     anchor="w", padx=8, pady=(6,2))
        tk.Frame(frame, bg="#00d4ff",
                 height=1).pack(fill="x", padx=8)
        return frame

    def build_gui(self):
        # ── Alert Banner (hidden by default) ──
        self.alert_banner = tk.Label(
            self.root,
            text="⚠ ALERT: Person has moved beyond 10 meters!",
            bg="#ff0000", fg="white",
            font=self.font_alert
        )

        # ── Main grid ──
        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(fill="both", expand=True, padx=10, pady=10)

        for i in range(3):
            main.columnconfigure(i, weight=1)
        for i in range(3):
            main.rowconfigure(i, weight=1)

        # ── CARD 1: Camera Feed ──
        cam_card = self.card(main, "📷 Live Camera", 0, 0, rowspan=2)
        self.cam_label = tk.Label(cam_card, bg="#0a0a1a")
        self.cam_label.pack(fill="both", expand=True, padx=8, pady=8)

        # ── CARD 2: Heart Rate ──
        hr_card = self.card(main, "❤ Heart Rate", 0, 1)
        self.hr_value = tk.Label(
            hr_card, text="---",
            bg="#16213e", fg="#ff4757",
            font=self.font_value
        )
        self.hr_value.pack(pady=5)
        tk.Label(hr_card, text="BPM",
                 bg="#16213e", fg="#ffffff",
                 font=self.font_label).pack()
        self.finger_status = tk.Label(
            hr_card, text="No finger detected",
            bg="#16213e", fg="#ffa502",
            font=self.font_small
        )
        self.finger_status.pack(pady=3)

        # ── CARD 3: SpO2 ──
        spo2_card = self.card(main, "🩸 Blood Oxygen (SpO2)", 0, 2)
        self.spo2_value = tk.Label(
            spo2_card, text="---",
            bg="#16213e", fg="#2ed573",
            font=self.font_value
        )
        self.spo2_value.pack(pady=5)
        tk.Label(spo2_card, text="%",
                 bg="#16213e", fg="#ffffff",
                 font=self.font_label).pack()
        self.spo2_status = tk.Label(
            spo2_card, text="Place finger on sensor",
            bg="#16213e", fg="#ffa502",
            font=self.font_small
        )
        self.spo2_status.pack(pady=3)

        # ── CARD 4: GPS ──
        gps_card = self.card(main, "📍 GPS Location", 1, 1)
        self.gps_lat = tk.Label(
            gps_card, text="Lat: ---",
            bg="#16213e", fg="#ffffff",
            font=self.font_label
        )
        self.gps_lat.pack(anchor="w", padx=10, pady=2)
        self.gps_lon = tk.Label(
            gps_card, text="Lon: ---",
            bg="#16213e", fg="#ffffff",
            font=self.font_label
        )
        self.gps_lon.pack(anchor="w", padx=10, pady=2)
        self.gps_dist = tk.Label(
            gps_card, text="Distance: 0.0 m",
            bg="#16213e", fg="#00d4ff",
            font=self.font_label
        )
        self.gps_dist.pack(anchor="w", padx=10, pady=2)
        self.gps_sats = tk.Label(
            gps_card, text="Satellites: ---",
            bg="#16213e", fg="#ffffff",
            font=self.font_small
        )
        self.gps_sats.pack(anchor="w", padx=10, pady=2)
        tk.Button(
            gps_card, text="Set Home Location",
            bg="#0f3460", fg="white",
            font=self.font_small,
            command=self.set_home
        ).pack(pady=5)

        # ── CARD 5: Recognition ──
        rec_card = self.card(main, "👤 Face & Voice Recognition", 1, 2)
        self.face_label = tk.Label(
            rec_card, text="Face:  ---",
            bg="#16213e", fg="#ffffff",
            font=self.font_label
        )
        self.face_label.pack(anchor="w", padx=10, pady=3)
        self.voice_label = tk.Label(
            rec_card, text="Voice: ---",
            bg="#16213e", fg="#ffffff",
            font=self.font_label
        )
        self.voice_label.pack(anchor="w", padx=10, pady=3)
        self.result_label = tk.Label(
            rec_card, text="---",
            bg="#16213e", fg="#ffa502",
            font=("Arial", 11, "bold")
        )
        self.result_label.pack(pady=3)

        # Buttons
        btn_frame = tk.Frame(rec_card, bg="#16213e")
        btn_frame.pack(fill="x", padx=8, pady=5)
        self.face_btn = tk.Button(
            btn_frame, text="FACE SCAN",
            bg="#0f3460", fg="white",
            font=self.font_small,
            command=self.start_face_scan
        )
        self.face_btn.pack(side="left", padx=3, expand=True, fill="x")
        self.voice_btn = tk.Button(
            btn_frame, text="VOICE SCAN",
            bg="#0f3460", fg="white",
            font=self.font_small,
            command=self.start_voice_scan,
            state="disabled"
        )
        self.voice_btn.pack(side="left", padx=3, expand=True, fill="x")

        # ── CARD 6: Status bar ──
        status_card = self.card(
            main, "System Status", 2, 0, colspan=3,
            color="#0f3460"
        )
        self.status_label = tk.Label(
            status_card,
            text="System ready — All sensors initializing...",
            bg="#0f3460", fg="#00d4ff",
            font=self.font_small
        )
        self.status_label.pack(pady=5)

    def set_home(self):
        if state["gps_fix"]:
            state["home_lat"] = state["lat"]
            state["home_lon"] = state["lon"]
            state["distance"] = 0.0
            state["alert"]    = False
            self.status_label.config(
                text="Home location updated to current position!")
        else:
            self.status_label.config(
                text="No GPS fix yet — move near a window!")

    def start_face_scan(self):
        if state["scanning"]:
            return
        self.face_btn.config(state="disabled",  text="Scanning...")
        self.voice_btn.config(state="disabled")
        self.result_label.config(
            text="Face scanning 10s...", fg="#ffa502")
        t = threading.Thread(
            target=self._face_scan_thread, daemon=True)
        t.start()

    def _face_scan_thread(self):
        # Count down 10 seconds
        for i in range(10, 0, -1):
            self.status_label.config(
                text="Face scanning... " + str(i) + "s remaining")
            time.sleep(1)

        if state["frame"] is not None:
            name, score = do_face_recognition(state["frame"])
            state["face_name"] = name
            color = "#2ed573" if name not in \
                    ["Unknown","No face"] else "#ff4757"
            self.face_label.config(
                text="Face:  " + name, fg=color)
            self.result_label.config(
                text="Face done! Click VOICE SCAN", fg="#ffa502")
            self.voice_btn.config(
                state="normal", text="VOICE SCAN")
        self.face_btn.config(
            state="normal", text="FACE SCAN")

    def start_voice_scan(self):
        if state["scanning"]:
            return
        self.voice_btn.config(
            state="disabled", text="Listening...")
        self.result_label.config(
            text="Speak now (5s)...", fg="#ffa502")
        t = threading.Thread(
            target=self._voice_scan_thread, daemon=True)
        t.start()

    def _voice_scan_thread(self):
        for i in range(5, 0, -1):
            self.status_label.config(
                text="Voice scanning... " + str(i) + "s remaining")
            time.sleep(1)

        name, conf = do_voice_recognition()
        state["voice_name"] = name
        confirmed = (
            state["face_name"] == name and
            name not in ["---","Unknown","No face","Error"]
        )
        state["confirmed"] = confirmed

        self.voice_label.config(
            text="Voice: " + name + " (" +
                 str(round(conf*100)) + "%)",
            fg="#2ed573" if confirmed else "#ff4757"
        )

        if confirmed:
            self.result_label.config(
                text="✓ CONFIRMED: " + name, fg="#2ed573")
        else:
            self.result_label.config(
                text="✗ NOT CONFIRMED", fg="#ff4757")

        self.voice_btn.config(
            state="normal", text="VOICE SCAN")
        self.status_label.config(
            text="Recognition complete!")

    def update_gui(self):
        """Update all GUI elements — runs every 500ms"""

        # ── Alert banner ──
        if state["alert"]:
            self.alert_banner.pack(fill="x")
        else:
            self.alert_banner.pack_forget()

        # ── Camera ──
        if state["frame"] is not None:
            frame = cv2.cvtColor(
                state["frame"], cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(frame)
            img   = img.resize((280, 210))
            photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=photo)
            self.cam_label.image = photo

        # ── Heart Rate ──
        hr = state["hr"]
        self.hr_value.config(
            text=str(hr) if hr > 0 else "---"
        )
        if state["finger_on"]:
            self.finger_status.config(
                text="Finger detected",
                fg="#2ed573"
            )
        else:
            self.finger_status.config(
                text="No finger detected",
                fg="#ffa502"
            )

        # ── SpO2 ──
        sp = state["spo2"]
        self.spo2_value.config(
            text=str(sp) if sp > 0 else "---"
        )
        if sp >= 95:
            self.spo2_status.config(
                text="Normal", fg="#2ed573")
        elif sp >= 90:
            self.spo2_status.config(
                text="Low - check again", fg="#ffa502")
        elif sp > 0:
            self.spo2_status.config(
                text="Very Low!", fg="#ff4757")
        else:
            self.spo2_status.config(
                text="Place finger on sensor", fg="#ffa502")

        # ── GPS ──
        if state["gps_fix"]:
            self.gps_lat.config(
                text="Lat: " + str(round(state["lat"], 6)))
            self.gps_lon.config(
                text="Lon: " + str(round(state["lon"], 6)))
            dist = state["distance"]
            self.gps_dist.config(
                text="Distance: " + str(round(dist, 1)) + " m",
                fg="#ff4757" if state["alert"] else "#00d4ff"
            )
            self.gps_sats.config(
                text="Satellites: " + str(state["satellites"]))
        else:
            self.gps_lat.config(text="Waiting for GPS fix...")
            self.gps_lon.config(text="Move near a window")

        # Schedule next update
        self.root.after(500, self.update_gui)

# ═══════════════════════════════════════
#              START APP
# ═══════════════════════════════════════
if __name__ == "__main__":
    # Start all background threads
    threads = [
        threading.Thread(target=health_thread, daemon=True),
        threading.Thread(target=gps_thread,    daemon=True),
        threading.Thread(target=camera_thread, daemon=True),
        threading.Thread(target=oled_thread,   daemon=True),
    ]
    for t in threads:
        t.start()

    print("Starting dashboard...")
    root = tk.Tk()
    app  = Dashboard(root)
    root.mainloop()