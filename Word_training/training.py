import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
def load_data(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path, header=None)
    
    # Separate features and labels
    # Assuming the last column is the label
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    return X, y

# Prepare the data
def prepare_data(X, y):
    # Use LabelEncoder to ensure continuous label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # One-hot encode the labels
    y_categorical = to_categorical(y_encoded)
    
    return X_scaled, y_categorical, scaler, label_encoder

# Build the neural network model
def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Visualize confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, label_names=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15,10))
    
    # Use actual label names if provided, otherwise use numeric labels
    if label_names is None:
        label_names = classes
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, 
                yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

# Main training script
def main():
    # Load data
    X, y = load_data('gesture_data.csv')
    
    # Print unique labels and their counts
    unique_labels, counts = np.unique(y, return_counts=True)
    print("Unique Labels:", unique_labels)
    print("Label Counts:", counts)
    
    # Prepare data
    X_scaled, y_encoded, scaler, label_encoder = prepare_data(X, y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    model = create_model(X_train.shape[1], y_encoded.shape[1])
    
    # Train the model
    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    
    # Predictions for confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Ensure unique labels match the actual number of classes
    unique_predicted_labels = np.unique(np.concatenate([y_true_classes, y_pred_classes]))
    
    # Create target names with original labels
    target_names = [str(label) for label in unique_predicted_labels]
    
    # Print classification report with corrected labels
    print("\nClassification Report:")
    print(classification_report(
        y_true_classes, 
        y_pred_classes, 
        labels=unique_predicted_labels,
        target_names=target_names
    ))
    
    # Plot confusion matrix with corrected labels
    plot_confusion_matrix(y_true_classes, y_pred_classes, unique_predicted_labels, target_names)
    
    # Plot training history
    plt.figure(figsize=(12,4))
    
    # Accuracy plot
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Save the model
    model.save('gesture_recognition_model.h5')
    
    # Save the scaler and label encoder
    from joblib import dump
    dump(scaler, 'gesture_scaler.joblib')
    dump(label_encoder, 'gesture_label_encoder.joblib')

if __name__ == '__main__':
    main()