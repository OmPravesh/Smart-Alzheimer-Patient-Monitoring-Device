# train_face.py (new version using deepface)
import os
import pickle
import cv2
import numpy as np
from deepface import DeepFace

print("Starting face training...")

known_embeddings = []
known_names = []
faces_dir = "dataset/faces"

for person_name in os.listdir(faces_dir):
    person_folder = os.path.join(faces_dir, person_name)
    if not os.path.isdir(person_folder):
        continue
    print(f"  Learning {person_name}'s face...")

    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        try:
            # Get face embedding (128 numbers describing the face)
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=False
            )
            known_embeddings.append(embedding[0]["embedding"])
            known_names.append(person_name)
        except Exception as e:
            print(f"    Skipping {img_file}: {e}")

# Save model
with open("face_model.pkl", "wb") as f:
    pickle.dump({"embeddings": known_embeddings, "names": known_names}, f)

print(f"\n✅ Face model trained and saved!")
print(f"   Learned {len(known_names)} face images total")