import os
import numpy as np
import cv2
import requests
import zipfile
import tarfile
import shutil
import argparse
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor

class DatasetDownloader:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        
        # Load or create label map
        self.label_map_path = os.path.join(output_dir, 'label_map.json')
        self.label_map = self._load_label_map()
    
    def _load_label_map(self) -> Dict[str, int]:
        """Load or create label mapping."""
        if os.path.exists(self.label_map_path):
            with open(self.label_map_path, 'r') as f:
                return json.load(f)
        
        # Default ASL alphabet mapping
        label_map = {
            chr(i): idx for idx, i in enumerate(range(65, 91))  # A-Z
        }
        
        # Save label map
        with open(self.label_map_path, 'w') as f:
            json.dump(label_map, f)
            
        return label_map
    
    def download_file(self, url: str, output_path: str):
        """Download a file from a URL with progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    def extract_archive(self, archive_path: str, extract_dir: str):
        """Extract a zip or tar archive."""
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            print(f"Unsupported archive format: {archive_path}")
    
    def process_image(self, image_path: str, label: str) -> Tuple[np.ndarray, bool]:
        """Process an image to extract hand landmarks."""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, False
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None, False
        
        # Extract hand landmarks
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.extend([landmark.x, landmark.y, landmark.z])
            landmarks.append(hand_data)
        
        return np.array(landmarks), True
    
    def process_directory(self, input_dir: str, label: str, output_dir: str):
        """Process all images in a directory for a specific label."""
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return
        
        print(f"Processing {len(image_files)} images for label '{label}'...")
        
        landmarks_list = []
        processed_count = 0
        
        for i, image_file in enumerate(tqdm(image_files)):
            image_path = os.path.join(input_dir, image_file)
            landmarks, success = self.process_image(image_path, label)
            
            if success:
                landmarks_list.append(landmarks)
                processed_count += 1
                
                # Save processed image
                output_path = os.path.join(label_dir, f"{i}.jpg")
                shutil.copy(image_path, output_path)
        
        # Save landmarks
        if landmarks_list:
            landmarks_array = np.array(landmarks_list)
            np.save(os.path.join(label_dir, "landmarks.npy"), landmarks_array)
            print(f"Saved {processed_count} samples for label '{label}'")
        else:
            print(f"No valid samples found for label '{label}'")
    
    def download_asl_alphabet_dataset(self):
        """Download and process the ASL Alphabet dataset."""
        dataset_url = "https://github.com/ardamavi/Sign-Language-Recognizer/raw/master/Dataset.zip"
        dataset_path = os.path.join(self.output_dir, "asl_alphabet.zip")
        
        # Download dataset
        if not os.path.exists(dataset_path):
            print("Downloading ASL Alphabet dataset...")
            self.download_file(dataset_url, dataset_path)
        
        # Extract dataset
        extract_dir = os.path.join(self.output_dir, "asl_alphabet")
        if not os.path.exists(extract_dir):
            print("Extracting dataset...")
            self.extract_archive(dataset_path, extract_dir)
        
        # Process dataset
        print("Processing dataset...")
        for label in self.label_map.keys():
            label_dir = os.path.join(extract_dir, label)
            if os.path.exists(label_dir):
                self.process_directory(label_dir, label, self.output_dir)
    
    def download_hand_gesture_dataset(self):
        """Download and process the Hand Gesture Recognition dataset."""
        dataset_url = "https://www.kaggle.com/datasets/gti-upm/leapgestrecog/download?datasetVersionNumber=1"
        dataset_path = os.path.join(self.output_dir, "hand_gesture.zip")
        
        # Download dataset
        if not os.path.exists(dataset_path):
            print("Downloading Hand Gesture dataset...")
            print("Note: This dataset requires Kaggle authentication.")
            print("Please download it manually from: https://www.kaggle.com/datasets/gti-upm/leapgestrecog")
            print(f"Save it to: {dataset_path}")
            input("Press Enter when done...")
        
        # Extract dataset
        extract_dir = os.path.join(self.output_dir, "hand_gesture")
        if not os.path.exists(extract_dir):
            print("Extracting dataset...")
            self.extract_archive(dataset_path, extract_dir)
        
        # Process dataset
        print("Processing dataset...")
        # Map dataset labels to our label map
        label_mapping = {
            "01_palm": "A",
            "02_l": "L",
            "03_fist": "F",
            "04_fist_moved": "F",
            "05_thumb": "T",
            "06_index": "I",
            "07_ok": "O",
            "08_palm_moved": "A",
            "09_c": "C",
            "10_down": "D"
        }
        
        for dataset_label, our_label in label_mapping.items():
            if our_label in self.label_map:
                label_dir = os.path.join(extract_dir, dataset_label)
                if os.path.exists(label_dir):
                    self.process_directory(label_dir, our_label, self.output_dir)
    
    def prepare_dataset(self):
        """Prepare the processed data for training."""
        print("Preparing dataset for training...")
        
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
        
        print(f"Dataset prepared: {len(X)} samples, {len(self.label_map)} classes")
        return X, y_one_hot

def main():
    parser = argparse.ArgumentParser(description='Download and prepare ASL datasets')
    parser.add_argument('--output_dir', type=str, default='backend/ml/data',
                        help='Directory to save processed data')
    parser.add_argument('--dataset', type=str, choices=['asl_alphabet', 'hand_gesture', 'all'],
                        default='asl_alphabet', help='Dataset to download and process')
    parser.add_argument('--prepare', action='store_true',
                        help='Prepare dataset for training after processing')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(output_dir=args.output_dir)
    
    if args.dataset == 'asl_alphabet' or args.dataset == 'all':
        downloader.download_asl_alphabet_dataset()
    
    if args.dataset == 'hand_gesture' or args.dataset == 'all':
        downloader.download_hand_gesture_dataset()
    
    if args.prepare:
        downloader.prepare_dataset()

if __name__ == '__main__':
    main() 