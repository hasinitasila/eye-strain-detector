import cv2
import mediapipe as mp
import time
import joblib
from face_mesh import FaceMeshDetector
from eye_utils import get_eye_points, compute_ear, LEFT_EYE, RIGHT_EYE
from blink_detector import BlinkDetector
from features import FeatureExtractor
from data_logger import DataLogger

detector = FaceMeshDetector()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

blink_detector = BlinkDetector(
    threshold=0.22,
    drop_threshold=0.03,
    min_frames=2,
    max_frames=6
)

# 60-second window
feature_extractor = FeatureExtractor(window_size=60)

logger = DataLogger()
current_label = None

# Counters for detection quality
total_frames = 0
detected_frames = 0

# Load model
model = joblib.load("model.pkl")

# Label mapping (define once)
label_map = {
    0: "REST 😴",
    1: "LOW 🙂",
    2: "MEDIUM 😐",
    3: "HIGH 😣"
}

# Store last prediction
last_prediction_text = "N/A"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks = detector.get_landmarks(rgb_frame)

    # Count total frames
    total_frames += 1

    if landmarks:
        detected_frames += 1

        for face_landmarks in landmarks:
            lm = face_landmarks.landmark

            # Eye points
            left_eye = get_eye_points(lm, LEFT_EYE, w, h)
            right_eye = get_eye_points(lm, RIGHT_EYE, w, h)

            # EAR
            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Blink detection
            blink_count = blink_detector.update(ear)

            # Feature extraction
            features = feature_extractor.update(
                ear,
                blink_count,
                detected_frames,
                total_frames
            )

            if features:
                print("Features:", features)

                # Reset counters
                total_frames = 0
                detected_frames = 0

                # Save data
                if current_label is not None:
                    if features["detection_rate"] >= 0.8 and features["ear_std"] < 0.1:
                        logger.log(features, current_label)
                        print(f"Saved with label {current_label} (good quality)")
                    else:
                        print("Skipped (low quality)")

                # Prediction
                X = [[
                    features["blink_rate"],
                    features["ear_mean"],
                    features["ear_std"],
                    features["session_time"]
                ]]

                prediction = model.predict(X)[0]
                last_prediction_text = label_map.get(prediction, "Unknown")

            # Draw eye points
            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            # Timer
            remaining_time = int(
                feature_extractor.window_size - (time.time() - feature_extractor.last_feature_time)
            )
            if remaining_time < 0:
                remaining_time = 0

            # Display
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Blinks: {blink_count}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.putText(frame, f"Window: {remaining_time}s", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.putText(frame, f"Prediction: {last_prediction_text}", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Eye Tracking", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        current_label = 1
        print("Label set: LOW")

    elif key == ord('2'):
        current_label = 2
        print("Label set: MEDIUM")

    elif key == ord('3'):
        current_label = 3
        print("Label set: HIGH")

    elif key == ord('0'):
        current_label = 0
        print("Label set: REST")

    elif key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()