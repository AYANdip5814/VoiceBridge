import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_VERSION = 'v1'
API_PREFIX = f'/api/{API_VERSION}'
HOST = os.getenv('HOST', 'localhost')
PORT = int(os.getenv('PORT', 5000))

# ML Model Configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'backend/ml/models')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.6))

# Video Configuration
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Security Configuration
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',') 