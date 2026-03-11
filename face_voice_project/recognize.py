# recognize_pi.py — Optimized for Raspberry Pi 4B
import cv2, pickle, os, time
import numpy as np
import sounddevice as sd
import librosa
import threading

print("Loading models...")

# Load face data
KNOWN_EMBEDDINGS = np.load("pi_face_embeddings.npy")
with open("pi_face_names.pkl", "rb") as f:
    KNOWN_NAMES = pickle.load(f)

# Load voice model
with open("voice_model.pkl", "rb") as f:
    voice_data = pickle.load(f)
voice_clf = voice_data["model"]
voice_le  = voice_data["encoder"]

print("Loaded", len(KNOWN_NAMES), "people:", KNOWN_NAMES)

# ── Face detector ──
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0
    return np.dot(a, b) / denom

def get_face_embedding_fast(gray_face):
    face_resized = cv2.resize(gray_face, (160, 160))
    hist = cv2.calcHist([face_resized], [0], None, [128], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# ---- BUILD HISTOGRAM CACHE on startup ----
# This replaces the pi_histograms folder approach
# We build histograms from the dataset/faces folder directly
print("Building face histogram cache...")
HIST_CACHE = {}
faces_dir = os.path.expanduser("~/face_voice_project/dataset/faces")

if os.path.exists(faces_dir):
    for person_name in os.listdir(faces_dir):
        person_folder = os.path.join(faces_dir, person_name)
        if not os.path.isdir(person_folder):
            continue
        hists = []
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # detect face in training image
            detected = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
            )
            if len(detected) > 0:
                x, y, w, h = detected[0]
                face_roi = gray[y:y+h, x:x+w]
            else:
                face_roi = gray  # use whole image if no face detected
            hist = get_face_embedding_fast(face_roi)
            hists.append(hist)
        if hists:
            # Average all histograms for this person
            HIST_CACHE[person_name] = np.mean(hists, axis=0)
            print("  Cached:", person_name)
else:
    print("  Warning: dataset/faces folder not found!")
    print("  Face recognition will return Unknown")

print("Face cache ready:", list(HIST_CACHE.keys()))

# ---- FACE RECOGNITION ----
def who_is_this_face(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            return "No face", 0

        # Take largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 255), 2)

        embedding = get_face_embedding_fast(face_roi)

        best_score = -1
        best_name  = "Unknown"

        for name, saved_hist in HIST_CACHE.items():
            score = cosine_similarity(embedding, saved_hist)
            if score > best_score:
                best_score = score
                best_name  = name

        if best_score > 0.7:
            return best_name, best_score
        else:
            return "Unknown", best_score

    except Exception as e:
        print("Face error:", e)
        return "No face", 0

# ---- VOICE RECOGNITION ----
def who_is_this_voice():
    DURATION = 5
    RATE     = 16000

    # Find USB Camera mic device number automatically
    mic_device = None
    for i, dev in enumerate(sd.query_devices()):
        if 'Camera' in dev['name'] or 'USB' in dev['name']:
            if dev['max_input_channels'] > 0:
                mic_device = i
                break

    print("Using mic device:", mic_device, "-",
          sd.query_devices(mic_device)['name'] if mic_device is not None else "default")

    # Record stereo (2 channels — what your mic supports)
    audio = sd.rec(
        int(DURATION * RATE),
        samplerate=RATE,
        channels=2,              # stereo
        dtype='float32',
        device=mic_device
    )
    sd.wait()

    # Convert stereo to mono
    audio = np.mean(audio, axis=1)

    try:
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        if len(audio_trimmed) < RATE:
            audio_trimmed = audio
    except Exception:
        audio_trimmed = audio

    mfcc      = librosa.feature.mfcc(y=audio_trimmed, sr=RATE, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std  = np.std(mfcc.T, axis=0)

    pitches, _ = librosa.piptrack(y=audio_trimmed, sr=RATE)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    pitch_std  = np.std(pitches[pitches > 0])  if np.any(pitches > 0) else 0

    rms      = librosa.feature.rms(y=audio_trimmed)
    rms_mean = np.mean(rms)
    rms_std  = np.std(rms)

    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_trimmed, sr=RATE))
    spec_rolloff  = np.mean(librosa.feature.spectral_rolloff(y=audio_trimmed,  sr=RATE))

    features = np.concatenate([
        mfcc_mean, mfcc_std,
        [pitch_mean, pitch_std, rms_mean, rms_std, spec_centroid, spec_rolloff]
    ]).reshape(1, -1)

    proba    = voice_clf.predict_proba(features)[0]
    best_idx = np.argmax(proba)
    name     = voice_le.inverse_transform([best_idx])[0]
    return name, proba[best_idx]

# ---- DRAW OVERLAY ----
def draw_overlay(frame, line1, line2="", color=(0, 255, 0), timer=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-120), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, line1, (20, h-80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if line2:
        cv2.putText(frame, line2, (20, h-45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if timer is not None:
        txt  = str(timer) + "s"
        size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)[0]
        cv2.putText(frame, txt, ((w-size[0])//2, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)
    return frame

def draw_progress_bar(frame, progress, color):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (20, h-20), (w-20, h-8), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h-20),
                  (int(20 + (w-40) * min(progress, 1.0)), h-8), color, -1)

# ── STATES ──
STATE_IDLE        = "idle"
STATE_FACE_SCAN   = "face_scan"
STATE_FACE_RESULT = "face_result"
STATE_VOICE_SCAN  = "voice_scan"
STATE_RESULT      = "result"

state           = STATE_IDLE
scan_start_time = 0
FACE_SCAN_DURATION  = 10
VOICE_SCAN_DURATION = 5

face_name = ""
face_score = 0.0
voice_name = ""
voice_conf = 0.0
best_face_frame = None
voice_result    = [None, None]

def run_voice():
    try:
        name, conf = who_is_this_voice()
        voice_result[0] = name
        voice_result[1] = conf
    except Exception as e:
        print("Voice error:", e)
        voice_result[0] = "Error"
        voice_result[1] = 0.0

# ---- CAMERA SETUP ----
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cam.set(cv2.CAP_PROP_FPS, 15)

if not cam.isOpened():
    print("ERROR: Camera not found! Check USB webcam connection.")
    exit()

print("Camera ready!")
print("Press SPACE to start | Q to quit")

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    now     = time.time()
    elapsed = now - scan_start_time
    display = frame.copy()

    # ── IDLE ──
    if state == STATE_IDLE:
        draw_overlay(display,
                     "Press SPACE to start",
                     "Face + Voice Recognition",
                     color=(0, 255, 0))

    # ── FACE SCANNING ──
    elif state == STATE_FACE_SCAN:
        remaining = max(0, FACE_SCAN_DURATION - int(elapsed))
        draw_overlay(display,
                     "STEP 1: Face Scanning...",
                     "Look at the camera",
                     color=(0, 200, 255), timer=remaining)
        draw_progress_bar(display, elapsed / FACE_SCAN_DURATION, (0, 200, 255))

        # Live face box
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 200, 255), 2)

        if elapsed >= FACE_SCAN_DURATION - 1:
            best_face_frame = frame.copy()

        if elapsed >= FACE_SCAN_DURATION:
            print("Analyzing face...")
            face_name, face_score = who_is_this_face(best_face_frame)
            print("Face:", face_name, "score:", round(face_score, 2))
            state = STATE_FACE_RESULT
            scan_start_time = time.time()

    # ── FACE RESULT ──
    elif state == STATE_FACE_RESULT:
        color = (0, 255, 0) if face_name not in ["Unknown", "No face"] else (0, 0, 255)
        blink_text = "Press SPACE for voice scan" if int(elapsed * 2) % 2 == 0 else ""
        draw_overlay(display,
                     "Face: " + face_name,
                     blink_text, color=color)

    # ── VOICE SCANNING ──
    elif state == STATE_VOICE_SCAN:
        remaining = max(0, VOICE_SCAN_DURATION - int(elapsed))
        draw_overlay(display,
                     "STEP 2: Voice Scanning...",
                     "Speak clearly into the mic",
                     color=(255, 165, 0), timer=remaining)
        draw_progress_bar(display, elapsed / VOICE_SCAN_DURATION, (255, 165, 0))

        if voice_result[0] is not None:
            voice_name = voice_result[0]
            voice_conf = voice_result[1]
            print("Voice:", voice_name, "conf:", round(voice_conf * 100), "%")
            state = STATE_RESULT
            scan_start_time = time.time()

    # ── FINAL RESULT ──
    elif state == STATE_RESULT:
        match = (face_name == voice_name and
                 face_name not in ["Unknown", "No face", "Error"])
        color       = (0, 255, 0) if match else (0, 0, 255)
        result_text = "CONFIRMED: " + face_name if match else "NOT CONFIRMED"

        draw_overlay(display, result_text,
                     "Press SPACE to scan again", color=color)

        h, w = display.shape[:2]
        cv2.putText(display, "Face: " + face_name,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(display, "Voice: " + voice_name + " " + str(round(voice_conf * 100)) + "%",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        if elapsed >= 8:
            state = STATE_IDLE

    cv2.imshow("Recognition - Pi", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if state == STATE_IDLE:
            scan_start_time = time.time()
            best_face_frame = None
            state = STATE_FACE_SCAN

        elif state == STATE_FACE_RESULT:
            voice_result = [None, None]
            t = threading.Thread(target=run_voice)
            t.daemon = True
            t.start()
            scan_start_time = time.time()
            state = STATE_VOICE_SCAN

        elif state == STATE_RESULT:
            scan_start_time = time.time()
            best_face_frame = None
            state = STATE_FACE_SCAN

    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()