import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp

class SignLanguageDataLoader:
    def __init__(self, data_path, max_frames=30, target_size=(224, 224)):
        """
        Initialize data loader for sign language videos
        
        Args:
            data_path (str): Root directory containing sign language videos
            max_frames (int): Maximum number of frames to extract
            target_size (tuple): Resize frame to this dimension
        """
        self.data_path = data_path
        self.max_frames = max_frames
        self.target_size = target_size
        
        # MediaPipe hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    def extract_hand_landmarks(self, frame):
        """
        Extract hand landmarks using MediaPipe
        
        Args:
            frame (numpy.ndarray): Input video frame
        
        Returns:
            numpy.ndarray: Normalized hand landmark features
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        # Default feature vector if no landmarks detected
        landmarks_features = np.zeros(63)  # 21 landmarks * 3 coordinates
        
        if results.multi_hand_landmarks:
            # Take the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract normalized landmark coordinates
            for i, landmark in enumerate(hand_landmarks.landmark):
                landmarks_features[i*3] = landmark.x
                landmarks_features[i*3 + 1] = landmark.y
                landmarks_features[i*3 + 2] = landmark.z
        
        return landmarks_features

    def load_video_features(self, video_path):
        """
        Load and process video features
        
        Args:
            video_path (str): Path to video file
        
        Returns:
            numpy.ndarray: Extracted features
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Prepare features array
        features = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, self.target_size)
            
            # Extract landmarks
            frame_features = self.extract_hand_landmarks(frame)
            features.append(frame_features)
            
            frame_count += 1
        
        cap.release()
        
        # Pad or truncate to consistent length
        if len(features) < self.max_frames:
            pad_length = self.max_frames - len(features)
            features.extend([np.zeros(63)] * pad_length)
        else:
            features = features[:self.max_frames]
        
        return np.array(features)

    def load_dataset(self):
        """
        Load entire dataset from directory
        
        Returns:
            tuple: (video_features, labels)
        """
        video_features = []
        labels = []
        
        # Check if data path exists
        if not os.path.exists(self.data_path):
            print(f"Error: Data path does not exist - {self.data_path}")
            return [], []
        
        # Traverse through subdirectories (each representing a sign)
        for sign_class in os.listdir(self.data_path):
            class_path = os.path.join(self.data_path, sign_class)
            
            # Skip if not a directory
            if not os.path.isdir(class_path):
                continue
            
            # Process each video in the class directory
            for video_file in os.listdir(class_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(class_path, video_file)
                    
                    try:
                        # Extract features
                        video_feature = self.load_video_features(video_path)
                        
                        # Add to dataset
                        video_features.append(video_feature)
                        labels.append(sign_class)
                    except Exception as e:
                        print(f"Error processing {video_path}: {e}")
        
        return video_features, labels

class SignLanguageTrainer:
    def __init__(self, sequence_length=30, feature_dim=63):
        """
        Initialize LSTM-based Sign Language Training Model
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.label_encoder = LabelEncoder()

    def build_model(self, num_classes):
        """
        Build LSTM Neural Network for sign language detection
        """
        self.model = Sequential([
            LSTM(256, return_sequences=True, 
                 input_shape=(self.sequence_length, self.feature_dim)),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(128, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

    def train(self, video_features, labels, epochs=50, batch_size=32):
        """
        Train the sign language detection model
        """
        # Convert to numpy arrays
        video_features = np.array(video_features)
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            video_features, 
            encoded_labels, 
            test_size=0.2, 
            random_state=42
        )
        
        # Build model based on number of classes
        self.build_model(len(np.unique(labels)))
        
        # Train model with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        # Fit the model
        history = self.model.fit(
            X_train, y_train, 
            validation_data=(X_test, y_test),
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[early_stopping]
        )
        
        return history

    def save_model(self, save_path='sign_language_model'):
        """
        Save trained model and label encoder classes
        """
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save(os.path.join(save_path, 'model.h5'))
        
        # Save label encoder classes
        np.save(os.path.join(save_path, 'label_classes.npy'), 
                self.label_encoder.classes_)

def main():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the dataset relative to the script location
    # Assumes dataset is in a 'dataset' folder next to the script
    data_path = os.path.join(current_dir, r"E:\datasets\Frames_Word_Level")
    
    # Print debug information
    print(f"Looking for dataset in: {data_path}")
    
    # Initialize data loader
    data_loader = SignLanguageDataLoader(data_path)
    
    # Load dataset
    video_features, labels = data_loader.load_dataset()
    
    # Check if data was loaded
    if not video_features:
        print("No videos found. Please check your dataset path and video files.")
        sys.exit(1)
    
    # Print dataset information
    print(f"Loaded {len(labels)} video samples")
    print(f"Unique classes: {set(labels)}")
    
    # Initialize trainer
    sign_trainer = SignLanguageTrainer()
    
    # Train model
    history = sign_trainer.train(video_features, labels)
    
    # Save trained model
    sign_trainer.save_model()

if __name__ == "__main__":
    main()