# train_voice.py (improved version)
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pickle, os

print("Starting voice training...")


def get_voice_features(file_path):
    """Extract richer voice features"""
    audio, sr = librosa.load(file_path, sr=16000)

    # Remove silence
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # 1. MFCC (voice tone fingerprint)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)  # ← NEW: variation info

    # 2. Pitch features
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    # 3. Energy/rhythm
    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # 4. Spectral features (voice brightness)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))

    # Combine all features into one array
    features = np.concatenate([
        mfcc_mean, mfcc_std,
        [pitch_mean, pitch_std, rms_mean, rms_std, spec_centroid, spec_rolloff]
    ])
    return features


X, y = [], []
voices_dir = "dataset/voices"

for person_name in os.listdir(voices_dir):
    person_folder = os.path.join(voices_dir, person_name)
    if not os.path.isdir(person_folder):
        continue
    print(f"  Learning {person_name}'s voice...")

    for audio_file in os.listdir(person_folder):
        file_path = os.path.join(person_folder, audio_file)
        try:
            features = get_voice_features(file_path)
            X.append(features)
            y.append(person_name)
        except Exception as e:
            print(f"    Skipping {audio_file}: {e}")

X = np.array(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Better pipeline: normalize + SVM with tuned settings
print("\n  Training improved classifier...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # normalize features
    ('svm', SVC(
        kernel='rbf',
        C=10,  # stronger classifier
        gamma='scale',
        probability=True
    ))
])
pipeline.fit(X, y_encoded)

# Test accuracy
if len(X) >= 4:
    scores = cross_val_score(pipeline, X, y_encoded, cv=min(4, len(X) // len(set(y))))
    print(f"  📊 Estimated accuracy: {scores.mean():.0%} (+/- {scores.std():.0%})")

with open("voice_model.pkl", "wb") as f:
    pickle.dump({"model": pipeline, "encoder": le}, f)

print(f"\n✅ Improved voice model saved!")
print(f"   Trained on {len(y)} clips across {len(set(y))} people")