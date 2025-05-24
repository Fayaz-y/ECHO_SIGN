import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from joblib import load
import subprocess

# Load trained model and scaler
model = tf.keras.models.load_model(r'D:\KPR_Hackathon\ECHO_SIGN\Word_training\gesture_recognition_model.h5')
scaler = load(r'D:\KPR_Hackathon\ECHO_SIGN\Word_training\gesture_scaler.joblib')

# MediaPipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Gesture Labels
label_map = {0: 'Help', 1: 'Normal'}

# Camera setup
cap = cv2.VideoCapture(0)
help_count = 0

def extract_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks), hand_landmarks
    return None, None

def predict_gesture(landmarks):
    landmarks_scaled = scaler.transform(landmarks.reshape(1, -1))
    prediction = model.predict(landmarks_scaled)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    return class_index, confidence

print("🖐️ Starting Gesture Recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    landmarks, hand_landmarks = extract_landmarks(frame)

    if landmarks is not None and hand_landmarks is not None:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        gesture_class, confidence = predict_gesture(landmarks)

        label = label_map.get(gesture_class, f'Gesture {gesture_class}')
        display_text = f'{label} (Confidence: {confidence:.2f})'

        if label == "Help":
            help_count += 1
        else:
            help_count = 0  # Reset if different gesture is detected

        if help_count == 3:
            print("🚨 Emergency Triggered! Running Script...")
            subprocess.Popen(["python", "code_sms.py"])
            break  # Stop processing

        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
