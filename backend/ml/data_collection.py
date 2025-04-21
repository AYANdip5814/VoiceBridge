import cv2
import numpy as np
import os
import time
import mediapipe as mp
import argparse
from tqdm import tqdm
import json
from typing import Dict, List, Tuple

class ISLDataCollector:
    def __init__(self, output_dir: str, num_samples: int = 100, delay: float = 0.5):
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.delay = delay
        
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
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load or create ISL label map
        self.label_map_path = os.path.join(output_dir, 'isl_label_map.json')
        self.label_map = self._load_label_map()
        
    def _load_label_map(self) -> Dict[str, int]:
        """Load or create ISL label mapping."""
        if os.path.exists(self.label_map_path):
            with open(self.label_map_path, 'r') as f:
                return json.load(f)
        
        # Comprehensive ISL signs mapping
        label_map = {
            # Basic Greetings and Courtesies
            "namaste": 0,
            "thank_you": 1,
            "please": 2,
            "yes": 3,
            "no": 4,
            "hello": 5,
            "goodbye": 6,
            "sorry": 7,
            "help": 8,
            "welcome": 9,
            
            # Basic Needs
            "water": 10,
            "food": 11,
            "bathroom": 12,
            "sleep": 13,
            "medicine": 14,
            "doctor": 15,
            "hospital": 16,
            
            # Emotions
            "happy": 17,
            "sad": 18,
            "angry": 19,
            "tired": 20,
            "sick": 21,
            "hungry": 22,
            "thirsty": 23,
            "scared": 24,
            "surprised": 25,
            
            # Time and Weather
            "today": 26,
            "tomorrow": 27,
            "yesterday": 28,
            "morning": 29,
            "afternoon": 30,
            "evening": 31,
            "night": 32,
            "hot": 33,
            "cold": 34,
            "rain": 35,
            "sunny": 36,
            
            # Family and Relationships
            "mother": 37,
            "father": 38,
            "sister": 39,
            "brother": 40,
            "grandmother": 41,
            "grandfather": 42,
            "friend": 43,
            "teacher": 44,
            
            # Numbers and Counting
            "one": 45,
            "two": 46,
            "three": 47,
            "four": 48,
            "five": 49,
            "six": 50,
            "seven": 51,
            "eight": 52,
            "nine": 53,
            "ten": 54,
            
            # Colors
            "red": 55,
            "blue": 56,
            "green": 57,
            "yellow": 58,
            "black": 59,
            "white": 60,
            
            # Common Actions
            "come": 61,
            "go": 62,
            "stop": 63,
            "wait": 64,
            "look": 65,
            "listen": 66,
            "speak": 67,
            "write": 68,
            "read": 69,
            "work": 70,
            
            # Questions
            "what": 71,
            "where": 72,
            "when": 73,
            "why": 74,
            "how": 75,
            "who": 76,
            
            # Places
            "home": 77,
            "school": 78,
            "market": 79,
            "office": 80,
            "hospital": 81,
            "temple": 82,
            "station": 83,
            
            # Transportation
            "bus": 84,
            "train": 85,
            "car": 86,
            "auto": 87,
            "walk": 88,
            "run": 89,
            
            # Food and Drink
            "rice": 90,
            "bread": 91,
            "milk": 92,
            "tea": 93,
            "coffee": 94,
            "fruit": 95,
            "vegetable": 96,
            
            # Emergency and Safety
            "emergency": 97,
            "police": 98,
            "fire": 99,
            "danger": 100,
            "safe": 101,
            "careful": 102
        }
        
        # Save label map
        with open(self.label_map_path, 'w') as f:
            json.dump(label_map, f)
            
        return label_map
    
    def collect_data(self, label: str):
        """Collect data for a specific ISL sign."""
        if label not in self.label_map:
            print(f"Label '{label}' not in label map. Adding it...")
            self.label_map[label] = len(self.label_map)
            with open(self.label_map_path, 'w') as f:
                json.dump(self.label_map, f)
        
        label_dir = os.path.join(self.output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print(f"Collecting data for ISL sign '{label}'...")
        print("Press 'q' to quit, 's' to skip current sample")
        
        samples_collected = 0
        landmarks_list = []
        
        with tqdm(total=self.num_samples) as pbar:
            while samples_collected < self.num_samples:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Flip the frame horizontally for a later selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Convert the BGR image to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame with MediaPipe
                hand_results = self.hands.process(frame_rgb)
                face_results = self.face.process(frame_rgb)
                
                # Draw hand and face landmarks
                annotated_frame = frame.copy()
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing = mp.solutions.drawing_utils
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        mp_drawing = mp.solutions.drawing_utils
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            face_landmarks,
                            self.mp_face.FACEMESH_CONTOURS
                        )
                
                # Display the frame
                cv2.putText(annotated_frame, f"ISL Sign: {label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Samples: {samples_collected}/{self.num_samples}", 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('ISL Data Collection', annotated_frame)
                
                # Wait for key press
                key = cv2.waitKey(1) & 0xFF
                
                # If 'q' is pressed, quit
                if key == ord('q'):
                    break
                
                # If 's' is pressed, skip current sample
                if key == ord('s'):
                    print("Skipping current sample")
                    continue
                
                # If space is pressed, collect sample
                if key == ord(' '):
                    if hand_results.multi_hand_landmarks or face_results.multi_face_landmarks:
                        # Extract hand and face landmarks
                        landmarks = []
                        
                        # Add hand landmarks
                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                hand_data = []
                                for landmark in hand_landmarks.landmark:
                                    hand_data.extend([landmark.x, landmark.y, landmark.z])
                                landmarks.append(hand_data)
                        
                        # Add face landmarks
                        if face_results.multi_face_landmarks:
                            for face_landmarks in face_results.multi_face_landmarks:
                                face_data = []
                                for landmark in face_landmarks.landmark:
                                    face_data.extend([landmark.x, landmark.y, landmark.z])
                                landmarks.append(face_data)
                        
                        # Save landmarks
                        landmarks_list.append(landmarks)
                        
                        # Save frame
                        frame_path = os.path.join(label_dir, f"{samples_collected}.jpg")
                        cv2.imwrite(frame_path, frame)
                        
                        samples_collected += 1
                        pbar.update(1)
                        
                        # Wait for a short delay
                        time.sleep(self.delay)
                    else:
                        print("No hand or face detected. Please show your hands and face clearly.")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Save landmarks to numpy file
        if landmarks_list:
            landmarks_array = np.array(landmarks_list)
            np.save(os.path.join(label_dir, "landmarks.npy"), landmarks_array)
            print(f"Saved {len(landmarks_list)} samples for ISL sign '{label}'")
        else:
            print(f"No samples collected for ISL sign '{label}'")
    
    def collect_all_labels(self):
        """Collect data for all ISL signs in the label map."""
        for label in self.label_map.keys():
            self.collect_data(label)
    
    def prepare_dataset(self):
        """Prepare the collected ISL data for training."""
        print("Preparing ISL dataset for training...")
        
        X = []
        y = []
        
        # Load data from each label directory
        for label, label_idx in self.label_map.items():
            label_dir = os.path.join(self.output_dir, label)
            if not os.path.exists(label_dir):
                print(f"Warning: Directory for label '{label}' does not exist")
                continue
            
            landmarks_path = os.path.join(label_dir, "landmarks.npy")
            if not os.path.exists(landmarks_path):
                print(f"Warning: No landmarks file found for label '{label}'")
                continue
            
            # Load landmarks
            landmarks = np.load(landmarks_path)
            
            # Add to dataset
            X.append(landmarks)
            y.extend([label_idx] * len(landmarks))
        
        if not X:
            print("Error: No data found to prepare dataset")
            return
        
        # Concatenate all landmarks
        X = np.concatenate(X, axis=0)
        y = np.array(y)
        
        # Convert labels to one-hot encoding
        y_one_hot = np.zeros((len(y), len(self.label_map)))
        for i, label_idx in enumerate(y):
            y_one_hot[i, label_idx] = 1
        
        # Save processed data
        np.save(os.path.join(self.output_dir, 'X_train.npy'), X)
        np.save(os.path.join(self.output_dir, 'y_train.npy'), y_one_hot)
        
        print(f"ISL Dataset prepared: {len(X)} samples, {len(self.label_map)} classes")
        return X, y_one_hot

def main():
    parser = argparse.ArgumentParser(description='Collect Indian Sign Language (ISL) data')
    parser.add_argument('--output_dir', type=str, default='backend/ml/data/isl',
                        help='Directory to save collected ISL data')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to collect per sign')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between samples in seconds')
    parser.add_argument('--label', type=str, default=None,
                        help='Specific ISL sign to collect data for (if None, collect for all)')
    parser.add_argument('--prepare', action='store_true',
                        help='Prepare dataset for training after collection')
    
    args = parser.parse_args()
    
    collector = ISLDataCollector(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        delay=args.delay
    )
    
    if args.label:
        collector.collect_data(args.label)
    else:
        collector.collect_all_labels()
    
    if args.prepare:
        collector.prepare_dataset()

if __name__ == '__main__':
    main() 