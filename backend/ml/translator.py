import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import time
import os
from typing import Dict, Optional
from .dataset import SignLanguageDataset
from backend.config import CONFIDENCE_THRESHOLD, MODEL_PATH

class SignLanguageTranslator:
    def __init__(self, model_type: str = 'cnn'):
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.model_path = os.path.join(MODEL_PATH, f'best_model.h5')
        self.model_type = model_type
        self.dataset = SignLanguageDataset(os.path.join(MODEL_PATH, '../data'))
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        
        # Load the translation model
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.label_map_inv = {v: k for k, v in self.dataset.label_map.items()}
        except Exception as e:
            print(f"Warning: Could not load model ({str(e)}). Using placeholder predictions.")
            self.model = None
            self.label_map_inv = {i: chr(i + 65) for i in range(26)}  # A-Z

    def preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess the frame for hand detection and recognition."""
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None
            
        # Extract hand landmarks
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.extend([landmark.x, landmark.y, landmark.z])
            landmarks.append(hand_data)
            
        # Preprocess landmarks
        landmarks = np.array(landmarks)
        landmarks = self.dataset.preprocess_landmarks(landmarks)
        
        if self.model_type == 'cnn':
            # Reshape for CNN input
            landmarks = landmarks.reshape((-1, 63, 3, 1))
        else:
            # Reshape for LSTM input
            landmarks = landmarks.reshape((1, -1))
            
        return landmarks

    def translate_frame(self, frame_data: np.ndarray) -> Dict:
        """Translate sign language from a video frame."""
        # Convert frame data to numpy array
        frame = np.array(frame_data, dtype=np.uint8)
        
        # Preprocess the frame
        landmarks = self.preprocess_frame(frame)
        if landmarks is None:
            return {
                'text': '',
                'confidence': 0.0,
                'timestamp': int(time.time() * 1000)
            }
            
        # Make prediction
        if self.model is not None:
            prediction = self.model.predict(landmarks, verbose=0)
            confidence = float(np.max(prediction))
            if confidence >= self.confidence_threshold:
                predicted_class = np.argmax(prediction)
                text = self.label_map_inv[predicted_class]
            else:
                text = ""
        else:
            # Placeholder prediction
            text = "Sample Sign"
            confidence = 0.8
            
        return {
            'text': text,
            'confidence': confidence,
            'timestamp': int(time.time() * 1000)
        }

    def get_visualization(self, frame: np.ndarray) -> np.ndarray:
        """Get visualization of hand landmarks on the frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            frame_rgb = frame.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_rgb,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return frame_rgb 