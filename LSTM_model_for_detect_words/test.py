import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

class SignLanguageTester:
    def __init__(self, model_path='sign_language_model'):
        """
        Initialize Sign Language Testing Model
        
        Args:
            model_path (str): Path to saved model and label encoder
        """
        # Load saved model
        self.model = tf.keras.models.load_model(
            os.path.join(model_path, 'model.h5')
        )
        
        # Load label encoder classes
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(
            os.path.join(model_path, 'label_classes.npy')
        )
        
        # Get model input shape
        self.sequence_length = self.model.input_shape[1]
        self.feature_dim = self.model.input_shape[2]

    def preprocess_sequence(self, input_sequence):
        """
        Preprocess input sequence for prediction
        
        Args:
            input_sequence (list): Input feature sequence
        
        Returns:
            numpy array: Padded and processed sequence
        """
        # Pad input sequence
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [input_sequence], 
            maxlen=self.sequence_length, 
            padding='post', 
            dtype='float32'
        )
        
        return padded_sequence

    def predict(self, input_sequence):
        """
        Predict sign language word from input feature sequence
        
        Args:
            input_sequence (list): Input feature sequence
        
        Returns:
            dict: Prediction results with class and probabilities
        """
        # Preprocess input sequence
        processed_sequence = self.preprocess_sequence(input_sequence)
        
        # Make prediction
        prediction = self.model.predict(processed_sequence)[0]
        
        # Get top predictions
        top_indices = prediction.argsort()[-3:][::-1]
        top_classes = self.label_encoder.inverse_transform(top_indices)
        top_probabilities = prediction[top_indices]
        
        # Prepare results
        results = {
            'top_prediction': top_classes[0],
            'predictions': [
                {
                    'class': cls, 
                    'probability': float(prob)
                } 
                for cls, prob in zip(top_classes, top_probabilities)
            ]
        }
        
        return results

    def evaluate(self, test_features, test_labels):
        """
        Evaluate model performance on test data
        
        Args:
            test_features (list): Test feature sequences
            test_labels (list): Corresponding test labels
        
        Returns:
            dict: Evaluation metrics
        """
        # Preprocess test data
        padded_features = tf.keras.preprocessing.sequence.pad_sequences(
            test_features, 
            maxlen=self.sequence_length, 
            padding='post', 
            dtype='float32'
        )
        
        # Encode labels
        encoded_labels = self.label_encoder.transform(test_labels)
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(
            padded_features, encoded_labels
        )
        
        # Detailed prediction analysis
        predictions = self.model.predict(padded_features)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Compute confusion matrix
        confusion_matrix = tf.math.confusion_matrix(
            encoded_labels, 
            predicted_classes
        ).numpy()
        
        return {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'confusion_matrix': confusion_matrix
        }

def load_test_data():
    """
    Load or generate test data
    Returns:
        tuple: test video features and corresponding labels
    """
    # In real scenario, replace with actual test data loading
    sign_words = ['hello', 'goodbye', 'thank you', 'yes', 'no']
    test_data = []
    test_labels = []

    # Generate sample test data
    for word in sign_words:
        # Simulate test video sequences
        for _ in range(10):
            # Simulate a video sequence
            video_sequence = [np.random.rand(128) for _ in range(np.random.randint(20, 40))]
            test_data.append(video_sequence)
            test_labels.append(word)

    return test_data, test_labels

def main():
    # Initialize tester
    sign_tester = SignLanguageTester()

    # Load test data
    test_features, test_labels = load_test_data()

    # Evaluate model
    evaluation_results = sign_tester.evaluate(test_features, test_labels)
    
    # Print evaluation results
    print("Evaluation Results:")
    print(f"Test Loss: {evaluation_results['loss']}")
    print(f"Test Accuracy: {evaluation_results['accuracy'] * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(evaluation_results['confusion_matrix'])

    # Example prediction
    sample_sequence = test_features[0]
    prediction_results = sign_tester.predict(sample_sequence)
    
    print("\nSample Prediction:")
    print(f"Top Prediction: {prediction_results['top_prediction']}")
    print("Top 3 Predictions:")
    for pred in prediction_results['predictions']:
        print(f"- {pred['class']}: {pred['probability'] * 100:.2f}%")

if __name__ == "__main__":
    main()