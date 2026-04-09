# Real-Time Eye Strain Detection System

A computer vision system that detects eye fatigue in real time using a webcam, MediaPipe facial landmarks, and machine learning — without any wearable sensors or special hardware.

---

## Motivation

Prolonged screen use is one of the most common causes of eye strain, affecting students, developers, and office workers worldwide. Most existing solutions require expensive hardware or are not personalized to individual users. This project explores a low-cost, non-invasive approach using only a standard webcam and computer vision.

---

## 📌 What It Does

- Tracks eye openness every frame using **Eye Aspect Ratio (EAR)**
- Detects blinks using a threshold + duration filter to avoid false positives
- Extracts features every 60-second window — blink rate, EAR mean, EAR std, session time
- Classifies fatigue into 4 levels: **REST / LOW / MEDIUM / HIGH**
- Displays live EAR, blink count, countdown timer, and prediction on screen
- Logs labeled data to CSV for model training

---

## How It Works

```
Webcam Frame
    ↓
MediaPipe FaceMesh (468 facial landmarks)
    ↓
Eye Landmark Extraction (6 points per eye)
    ↓
EAR Calculation → Blink Detection
    ↓
60-Second Feature Window
    ↓
Random Forest Classifier
    ↓
Fatigue Label (REST / LOW / MEDIUM / HIGH)
```

---

## 📁 Project Structure

```
├── main.py            # Main loop — webcam, display, prediction
├── face_mesh.py       # MediaPipe FaceMesh setup
├── eye_utils.py       # EAR formula + eye landmark extraction
├── blink_detector.py  # Blink counting logic with false-positive filtering
├── features.py        # 60-second windowed feature extraction
├── data_logger.py     # CSV logging for labeled data collection
├── train_model.py     # Random Forest training + evaluation
└── requirements.txt   # Dependencies
```

---

## Tech Stack

- Python
- OpenCV
- MediaPipe
- scikit-learn (Random Forest)
- NumPy, pandas, joblib

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Collect labeled data
# Run the app and press 0/1/2/3 to label your current fatigue level
python main.py

# Step 2: Train the model
python train_model.py

# Step 3: Run with live predictions
python main.py
```

---

## Features Used for Classification

| Feature | Description |
|---|---|
| `blink_rate` | Blinks per minute in the 60s window |
| `ear_mean` | Average eye openness across the window |
| `ear_std` | Variability in eye openness — lower std indicates fatigue |
| `session_time` | Total elapsed time since session started |

---

## 🏷️ Fatigue Labels

| Label | Meaning |
|---|---|
| 0 — REST | Eyes closed or resting |
| 1 — LOW | Minimal strain |
| 2 — MEDIUM | Moderate fatigue |
| 3 — HIGH | High eye strain |

---

## Known Limitations

- EAR threshold (0.22) is hardcoded — does not adapt to individual eye geometry
- Model trained on self-collected single-user data — generalization across users is limited
- No gaze tracking — cannot detect if user is looking away from the screen
- Sensitive to lighting conditions and camera angle

---

## Planned Improvements

These limitations directly motivate the next version of this system:

- **Personalized calibration** — compute EAR threshold per user during a 30-second baseline phase, replacing the hardcoded value
- **Iris-based gaze tracking** — use MediaPipe iris landmarks to detect gaze deviation from screen center
- **PERCLOS** — add percentage of eye closure time as a feature, a clinically validated fatigue metric
- **Mean blink duration** — longer individual blinks correlate with higher fatigue levels
- **Multi-user dataset** — validate across multiple users to improve model generalization

---
