import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from joblib import load
import subprocess



# Load the trained model and scaler
model = tf.keras.models.load_model(r'E:\FINAL-SIGN\Word_training\gesture_recognition_model.h5')
scaler = load(r'E:\FINAL-SIGN\Word_training\gesture_scaler.joblib')

def runing():
    subprocess.Popen(["python", "drop4.py"])
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Dictionary to map labels (adjust based on your training data)
label_map = {
    0: 'Help', 1:'Normal'
}


def extract_landmarks(frame):
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = hands.process(rgb_frame)
    
    # If hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks), hand_landmarks
    
    return None, None

def predict_gesture(landmarks):
    # Reshape and scale the landmarks
    landmarks_scaled = scaler.transform(landmarks.reshape(1, -1))
    
    # Predict
    prediction = model.predict(landmarks_scaled)
    
    # Get the class with highest probability
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    
    return class_index, confidence

def main():
    c = 0
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Extract landmarks
        landmarks, hand_landmarks = extract_landmarks(frame)
        
        if landmarks is not None and hand_landmarks is not None:
            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            # Predict gesture
            gesture_class, confidence = predict_gesture(landmarks)
            
            # Display prediction
            label = label_map.get(gesture_class, f'Gesture {gesture_class}')
            display_text = f'{label} (Confidence: {confidence:.2f})'
            
            if label == "Help":
                c+=1
            if c == 3:
                runing()
                return 0 
                
            # Put text on frame
            cv2.putText(
                frame, 
                display_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
        
        # Display the frame
        cv2.imshow('Gesture Recognition', frame)
        
        # Break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()