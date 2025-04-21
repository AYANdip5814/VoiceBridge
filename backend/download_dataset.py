import os
import requests
import zipfile
import shutil
from tqdm import tqdm

def download_file(url, filename):
    """
    Download a file from a URL with a progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def prepare_dataset():
    """
    Download and prepare the sign language dataset
    """
    # Create necessary directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '../data')
    dataset_dir = os.path.join(data_dir, 'sign_language_dataset')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download datasets
    print("Downloading sign language datasets...")
    
    datasets = [
        {
            'name': 'ASL Alphabet',
            'url': 'https://www.kaggle.com/datasets/grassknoted/asl-alphabet/download',
            'filename': 'asl_alphabet.zip'
        },
        {
            'name': 'Sign Language MNIST',
            'url': 'https://www.kaggle.com/datasets/datamunge/sign-language-mnist/download',
            'filename': 'sign_mnist.zip'
        }
    ]
    
    for dataset in datasets:
        zip_path = os.path.join(data_dir, dataset['filename'])
        print(f"\nDownloading {dataset['name']}...")
        print("Note: You'll need to manually download the dataset from Kaggle:")
        print(f"1. Visit: {dataset['url']}")
        print("2. Sign in to Kaggle")
        print(f"3. Download and place the file in: {zip_path}")
        input("Press Enter once you've downloaded the file...")
        
        if os.path.exists(zip_path):
            print(f"Extracting {dataset['filename']}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            os.remove(zip_path)
        else:
            print(f"Warning: {dataset['filename']} not found. Skipping...")
    
    # Organize the dataset
    print("\nOrganizing dataset...")
    
    # Create gesture folders
    gestures = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing'
    ]
    
    for gesture in gestures:
        os.makedirs(os.path.join(dataset_dir, gesture), exist_ok=True)
    
    # Move files to appropriate folders
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                for gesture in gestures:
                    if gesture.lower() in file.lower():
                        src = os.path.join(root, file)
                        dst = os.path.join(dataset_dir, gesture, file)
                        shutil.move(src, dst)
    
    # Clean up temporary directories
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path) and item not in gestures:
            shutil.rmtree(item_path)
    
    print("\nDataset preparation complete!")
    print(f"Dataset location: {dataset_dir}")
    print("Number of gestures:", len(gestures))
    
    # Print statistics
    print("\nDataset statistics:")
    for gesture in gestures:
        gesture_dir = os.path.join(dataset_dir, gesture)
        if os.path.exists(gesture_dir):
            num_files = len([f for f in os.listdir(gesture_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{gesture}: {num_files} images")

if __name__ == "__main__":
    prepare_dataset() 