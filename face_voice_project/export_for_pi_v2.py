# save as: export_for_pi_v2.py
import cv2
import pickle
import os
import numpy as np

print("Building LBPH face model for Raspberry Pi...")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# LBPH recognizer — fast and accurate, no TensorFlow needed
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces_data  = []
labels      = []
label_names = {}
label_id    = 0

faces_dir = "dataset/faces"

for person_name in sorted(os.listdir(faces_dir)):
    person_folder = os.path.join(faces_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"  Processing {person_name} (ID={label_id})...")
    label_names[label_id] = person_name
    count = 0

    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try to detect face
        detected = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )

        if len(detected) > 0:
            x, y, w, h = detected[0]
            face_roi = gray[y:y+h, x:x+w]
        else:
            # No face detected — use center crop
            h, w = gray.shape
            margin = min(h, w) // 4
            face_roi = gray[margin:h-margin, margin:w-margin]

        # Resize to standard size
        face_roi = cv2.resize(face_roi, (100, 100))
        faces_data.append(face_roi)
        labels.append(label_id)
        count += 1

    print(f"    Added {count} images for {person_name}")
    label_id += 1

# Train LBPH
print("\nTraining LBPH recognizer...")
recognizer.train(faces_data, np.array(labels))

# Save model
recognizer.save("lbph_face_model.yml")
with open("lbph_label_names.pkl", "wb") as f:
    pickle.dump(label_names, f)

print("\nDone! Created:")
print("  lbph_face_model.yml")
print("  lbph_label_names.pkl")
print(f"  Trained on {len(faces_data)} face images")
print(f"  People: {label_names}")