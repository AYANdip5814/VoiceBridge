import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def create_model(input_shape, num_classes):
    """
    Create a more sophisticated model for better gesture classification
    """
    model = models.Sequential([
        # Input layer
        layers.Dense(256, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Hidden layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile with a lower learning rate for better convergence
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def augment_landmarks(landmarks, noise_factor=0.05):
    """
    Apply data augmentation to landmarks by adding random noise
    """
    augmented = landmarks.copy()
    noise = np.random.normal(0, noise_factor, augmented.shape)
    augmented += noise
    return augmented

def prepare_dataset(data_dir, augment=True):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
    
    X = []
    y = []
    
    # Process each gesture folder
    for gesture_id, gesture_folder in enumerate(os.listdir(data_dir)):
        gesture_path = os.path.join(data_dir, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue
            
        print(f"Processing gesture: {gesture_folder}")
        
        # Process each image in the gesture folder
        for img_file in os.listdir(gesture_path):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process image with MediaPipe
            results = hands.process(rgb_img)
            
            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Pad or truncate landmarks to fixed size
                max_landmarks = 63  # 21 landmarks * 3 coordinates
                if len(landmarks) < max_landmarks:
                    landmarks.extend([0] * (max_landmarks - len(landmarks)))
                elif len(landmarks) > max_landmarks:
                    landmarks = landmarks[:max_landmarks]
                
                X.append(landmarks)
                y.append(gesture_id)
                
                # Data augmentation
                if augment:
                    for _ in range(3):  # Create 3 augmented samples per image
                        augmented_landmarks = augment_landmarks(landmarks)
                        X.append(augmented_landmarks)
                        y.append(gesture_id)
    
    return np.array(X), np.array(y)

def train_sign_language_model():
    # Set paths
    data_dir = os.path.join(os.path.dirname(__file__), '../data/sign_language_dataset')
    model_save_path = os.path.join(os.path.dirname(__file__), '../models/sign_language_model.h5')
    
    # Prepare dataset
    print("Preparing dataset...")
    X, y = prepare_dataset(data_dir, augment=True)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and train model
    print("Training model...")
    num_classes = len(np.unique(y))
    model = create_model((X.shape[1],), num_classes)
    
    # Define callbacks for better training
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    # Train with class weights to handle imbalanced data
    class_weights = {}
    for i in range(num_classes):
        class_weights[i] = 1.0 / np.sum(y_train == i)
    
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    # Print classification report
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    np.save(os.path.join(os.path.dirname(__file__), '../models/confusion_matrix.npy'), cm)
    print("\nConfusion matrix saved to: confusion_matrix.npy")

if __name__ == "__main__":
    train_sign_language_model() 