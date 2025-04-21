import numpy as np
import tensorflow as tf
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

class ISLInterpreter:
    def __init__(
        self,
        model_path: str = 'models/isl',
        model_type: str = 'cnn',
        confidence_threshold: float = 0.5,
        use_face_features: bool = True
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.use_face_features = use_face_features
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.face = self.mp_face.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.7
        )
        
        # Load model and label map
        self.model = self._load_model()
        self.label_map = self._load_label_map()
        
        # Initialize interpretation history
        self.interpretation_history: List[Dict] = []
        self.current_sign: Optional[str] = None
        self.current_confidence: float = 0.0
    
    def _load_model(self) -> tf.keras.Model:
        """Load the trained ISL model."""
        model_file = os.path.join(self.model_path, f'{self.model_type}_best.h5')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        return tf.keras.models.load_model(model_file)
    
    def _load_label_map(self) -> Dict[int, str]:
        """Load the ISL label map."""
        label_map_file = os.path.join(self.model_path, 'isl_label_map.json')
        if not os.path.exists(label_map_file):
            raise FileNotFoundError(f"Label map file not found: {label_map_file}")
        
        with open(label_map_file, 'r') as f:
            label_map = json.load(f)
        
        # Convert string keys to integers
        return {int(k): v for k, v in label_map.items()}
    
    def _preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Preprocess a frame to extract hand and face landmarks.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Tuple of (processed features, success flag)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        hand_results = self.hands.process(frame_rgb)
        face_results = self.face.process(frame_rgb)
        
        if not (hand_results.multi_hand_landmarks or (self.use_face_features and face_results.multi_face_landmarks)):
            return None, False
        
        # Extract landmarks
        features = []
        
        # Add hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_data = []
                for landmark in hand_landmarks.landmark:
                    hand_data.extend([landmark.x, landmark.y, landmark.z])
                features.append(hand_data)
        
        # Add face landmarks
        if self.use_face_features and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                face_data = []
                for landmark in face_landmarks.landmark:
                    face_data.extend([landmark.x, landmark.y, landmark.z])
                features.append(face_data)
        
        # Convert to numpy array and reshape
        features = np.array(features)
        if self.model_type == 'cnn':
            features = features.reshape(1, -1, 3, 1)
        else:
            features = features.reshape(1, -1, features.shape[-1])
        
        return features, True
    
    def interpret_frame(self, frame: np.ndarray) -> Dict:
        """
        Interpret a single frame to detect ISL signs.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Dictionary containing interpretation results
        """
        # Preprocess frame
        features, success = self._preprocess_frame(frame)
        if not success:
            return {
                'sign': None,
                'confidence': 0.0,
                'timestamp': datetime.now().timestamp()
            }
        
        # Get model predictions
        predictions = self.model.predict(features, verbose=0)
        confidence = float(np.max(predictions))
        sign_idx = int(np.argmax(predictions))
        
        # Get sign label
        sign = self.label_map.get(sign_idx, 'unknown')
        
        # Update current state
        self.current_sign = sign if confidence >= self.confidence_threshold else None
        self.current_confidence = confidence
        
        # Add to history
        result = {
            'sign': sign,
            'confidence': confidence,
            'timestamp': datetime.now().timestamp()
        }
        self.interpretation_history.append(result)
        
        # Keep only last 100 interpretations
        if len(self.interpretation_history) > 100:
            self.interpretation_history = self.interpretation_history[-100:]
        
        return result
    
    def get_current_state(self) -> Dict:
        """Get the current interpretation state."""
        return {
            'current_sign': self.current_sign,
            'current_confidence': self.current_confidence,
            'recent_interpretations': self.interpretation_history[-10:] if self.interpretation_history else []
        }
    
    def reset(self):
        """Reset the interpreter state."""
        self.interpretation_history = []
        self.current_sign = None
        self.current_confidence = 0.0 