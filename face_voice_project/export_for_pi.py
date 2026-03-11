# save as: export_for_pi.py
# Run this on your PC, then copy output files to Pi

import pickle
import numpy as np
import os
from deepface import DeepFace

print("Exporting lightweight face data for Raspberry Pi...")

known_embeddings = []
known_names = []
faces_dir = "dataset/faces"

for person_name in os.listdir(faces_dir):
    person_folder = os.path.join(faces_dir, person_name)
    if not os.path.isdir(person_folder):
        continue
    print(f"  Processing {person_name}...")

    person_embeddings = []
    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="opencv"
            )
            person_embeddings.append(result[0]["embedding"])
        except Exception as e:
            print(f"    Skipping {img_file}: {e}")

    if person_embeddings:
        # Save AVERAGE embedding per person (faster + lighter)
        avg_embedding = np.mean(person_embeddings, axis=0)
        known_embeddings.append(avg_embedding)
        known_names.append(person_name)
        print(f"    ✅ {person_name}: averaged {len(person_embeddings)} images")

# Save as lightweight numpy format
np.save("pi_face_embeddings.npy", np.array(known_embeddings))
with open("pi_face_names.pkl", "wb") as f:
    pickle.dump(known_names, f)

print(f"\n✅ Done! Created:")
print(f"   pi_face_embeddings.npy  ← copy this to Pi")
print(f"   pi_face_names.pkl       ← copy this to Pi")