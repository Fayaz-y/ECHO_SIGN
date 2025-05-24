import cv2
import numpy as np
import mediapipe as mp
import csv
import os

# Use raw string or forward slashes for path to avoid Windows path issues
root_directory = r'E:\FINAL-SIGN\Word_training\datasets'  # Use raw string or use forward slashes
# Alternatively: root_directory = 'E:/sign/train'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Verify the directory exists
if not os.path.exists(root_directory):
    print(f"Error: Directory {root_directory} does not exist!")
    exit(1)

# Open CSV file to store data
file = open('gesture_data.csv', 'a', newline='')
csv_writer = csv.writer(file)

# Dictionary to map labels to numeric indices
label_map = {}
current_label_index = 0

# Process each gesture subdirectory
for gesture_dir in os.listdir(root_directory):
    # Full path to the gesture directory
    gesture_path = os.path.join(root_directory, gesture_dir)
    
    # Skip if not a directory
    if not os.path.isdir(gesture_path):
        continue
    
    # Assign or retrieve numeric label for this gesture
    if gesture_dir not in label_map:
        label_map[gesture_dir] = current_label_index
        current_label_index += 1
    
    label = label_map[gesture_dir]
    
    # Get list of image files in this gesture directory
    image_files = [f for f in os.listdir(gesture_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # Process each image in the gesture directory
    for image_filename in image_files:
        # Full path to the image
        image_path = os.path.join(gesture_path, image_filename)
        
        # Read the image
        frame = cv2.imread(image_path)
        
        # Check if image is loaded successfully
        if frame is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data = []
                for landmark in hand_landmarks.landmark:
                    data.extend([landmark.x, landmark.y, landmark.z])
                
                # Add the label
                data.append(label)
                
                # Write data to CSV
                csv_writer.writerow(data)
                
                # Optional: Visualize landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imshow(f"Hand Tracking - {gesture_dir}", frame)
                cv2.waitKey(500)  # Pause to show each image

# Print out the label mapping
print("Label Mapping:")
for gesture, index in label_map.items():
    print(f"{gesture}: {index}")

# Close everything
cv2.destroyAllWindows()
file.close()

print("Image processing complete.")