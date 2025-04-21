import pyttsx3  # For Text-to-Speech
try:
    import mediapipe as mp  # For gesture recognition
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MediaPipe import failed: {e}")
    print("Continuing without MediaPipe functionality")
    MEDIAPIPE_AVAILABLE = False
import speech_recognition as sr  # For Speech-to-Text
from flask import Flask, request, jsonify, Response, render_template
import cv2
import numpy as np
from flask_cors import CORS
import base64
import tensorflow as tf
from PIL import Image
import io
import os
import json
import uuid
from transformers import AutoTokenizer, AutoModelForSeq2Seq
from fer import FER
from deepface import DeepFace
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from gesture_mappings import get_gesture_mapping, get_phrase_sequence, COMMON_PHRASES
from utils.obs_camera import OBSCamera
import threading
import time
import hashlib
import datetime

app = Flask(__name__)
CORS(app)

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize MediaPipe Hands for Gesture Recognition
if MEDIAPIPE_AVAILABLE:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    # Initialize MediaPipe Face Mesh for facial landmarks
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5
    )
else:
    print("MediaPipe not available - gesture recognition will be disabled")

# Initialize emotion detector
emotion_detector = FER(mtcnn=True)

# Dictionary to store language-specific models
sign_models = {}
translation_models = {}
tokenizers = {}

# WebRTC signaling - store active calls
active_calls = {}

# Global variables for OBS camera
obs_camera = None
frame_lock = threading.Lock()
current_frame = None
is_processing = False

# User management
users = {}  # In-memory user storage (replace with database in production)
active_users = {}  # Track online users

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user."""
    return stored_password == hash_password(provided_password)

@app.route('/api/users/register', methods=['POST'])
def register_user():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    if username in users:
        return jsonify({'error': 'Username already exists'}), 409
    
    users[username] = {
        'password': hash_password(password),
        'email': email,
        'created_at': datetime.datetime.now().isoformat(),
        'last_login': None
    }
    
    return jsonify({'message': 'User registered successfully', 'username': username}), 201

@app.route('/api/users/login', methods=['POST'])
def login_user():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    if username not in users:
        return jsonify({'error': 'User not found'}), 404
    
    if not verify_password(users[username]['password'], password):
        return jsonify({'error': 'Invalid password'}), 401
    
    # Update last login time
    users[username]['last_login'] = datetime.datetime.now().isoformat()
    
    # Add to active users
    active_users[username] = {
        'status': 'online',
        'last_seen': datetime.datetime.now().isoformat()
    }
    
    return jsonify({
        'message': 'Login successful',
        'username': username
    }), 200

@app.route('/api/users/logout', methods=['POST'])
def logout_user():
    data = request.json
    username = data.get('username')
    
    if not username:
        return jsonify({'error': 'Username is required'}), 400
    
    if username in active_users:
        del active_users[username]
    
    return jsonify({'message': 'Logout successful'}), 200

@app.route('/api/users/status', methods=['GET'])
def get_user_status():
    username = request.args.get('username')
    
    if not username:
        return jsonify({'error': 'Username is required'}), 400
    
    if username not in users:
        return jsonify({'error': 'User not found'}), 404
    
    status = 'offline'
    if username in active_users:
        status = active_users[username]['status']
    
    return jsonify({
        'username': username,
        'status': status,
        'last_seen': active_users.get(username, {}).get('last_seen')
    }), 200

@app.route('/api/users/list', methods=['GET'])
def list_users():
    # Return list of all users with their status
    user_list = []
    for username in users:
        status = 'offline'
        last_seen = None
        if username in active_users:
            status = active_users[username]['status']
            last_seen = active_users[username]['last_seen']
        
        user_list.append({
            'username': username,
            'status': status,
            'last_seen': last_seen
        })
    
    return jsonify({'users': user_list}), 200

# Load models on startup
def load_models():
    # ASL model (default)
    asl_model_path = os.path.join(os.path.dirname(__file__), '../models/asl_model.h5')
    if os.path.exists(asl_model_path):
        sign_models['asl'] = tf.keras.models.load_model(asl_model_path)
    else:
        print("Warning: ASL model not found. Using default model.")
        sign_models['asl'] = None
    
    # BSL model
    bsl_model_path = os.path.join(os.path.dirname(__file__), '../models/bsl_model.h5')
    if os.path.exists(bsl_model_path):
        sign_models['bsl'] = tf.keras.models.load_model(bsl_model_path)
    else:
        print("Warning: BSL model not found. Using ASL model as fallback.")
        sign_models['bsl'] = sign_models['asl']
    
    # ISL model
    isl_model_path = os.path.join(os.path.dirname(__file__), '../models/isl_model.h5')
    if os.path.exists(isl_model_path):
        sign_models['isl'] = tf.keras.models.load_model(isl_model_path)
    else:
        print("Warning: ISL model not found. Using ASL model as fallback.")
        sign_models['isl'] = sign_models['asl']
    
    # Auslan model
    auslan_model_path = os.path.join(os.path.dirname(__file__), '../models/auslan_model.h5')
    if os.path.exists(auslan_model_path):
        sign_models['auslan'] = tf.keras.models.load_model(auslan_model_path)
    else:
        print("Warning: Auslan model not found. Using ASL model as fallback.")
        sign_models['auslan'] = sign_models['asl']
    
    # Load translation models for each language
    try:
        # ASL translation model
        tokenizers['asl'] = AutoTokenizer.from_pretrained("sign-language-translation/asl-to-text")
        translation_models['asl'] = AutoModelForSeq2Seq.from_pretrained("sign-language-translation/asl-to-text")
        
        # BSL translation model
        tokenizers['bsl'] = AutoTokenizer.from_pretrained("sign-language-translation/bsl-to-text")
        translation_models['bsl'] = AutoModelForSeq2Seq.from_pretrained("sign-language-translation/bsl-to-text")
        
        # ISL translation model
        tokenizers['isl'] = AutoTokenizer.from_pretrained("sign-language-translation/isl-to-text")
        translation_models['isl'] = AutoModelForSeq2Seq.from_pretrained("sign-language-translation/isl-to-text")
        
        # Auslan translation model
        tokenizers['auslan'] = AutoTokenizer.from_pretrained("sign-language-translation/auslan-to-text")
        translation_models['auslan'] = AutoModelForSeq2Seq.from_pretrained("sign-language-translation/auslan-to-text")
    except Exception as e:
        print(f"Error loading translation models: {e}")
        print("Using default ASL translation model for all languages")
        # Use ASL model as fallback for all languages
        for lang in ['bsl', 'isl', 'auslan']:
            tokenizers[lang] = tokenizers['asl']
            translation_models[lang] = translation_models['asl']

# Initialize TTS engine
engine = pyttsx3.init()

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Load models on startup
load_models()

def process_frame(frame, language='asl'):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize result dictionary
    result = {
        "gesture": "unknown",
        "interpretation": "No gesture detected",
        "emoji": "❓",
        "emotion": "neutral",
        "emotion_confidence": 0.0,
        "landmarks": []
    }
    
    # Process hand gestures if detected and MediaPipe is available
    if MEDIAPIPE_AVAILABLE:
        # Process the frame with MediaPipe for hand detection
        hand_results = hands.process(rgb_frame)
        
        # Process the frame with MediaPipe for face detection
        face_results = face_mesh.process(rgb_frame)
        
        # Process hand gestures if detected
        if hand_results.multi_hand_landmarks:
            # Extract hand landmarks
            landmarks = []
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Store landmarks for animation
            result["landmarks"] = landmarks
            
            # Normalize landmarks
            landmarks_array = np.array(landmarks).reshape(1, -1)
            
            # Get the appropriate model for the selected language
            sign_model = sign_models.get(language, sign_models['asl'])
            
            if sign_model:
                # Get prediction from the model
                prediction = sign_model.predict(landmarks_array)
                gesture_index = np.argmax(prediction[0])
                
                # Get the appropriate translation model and tokenizer
                tokenizer = tokenizers.get(language, tokenizers['asl'])
                translation_model = translation_models.get(language, translation_models['asl'])
                
                # Convert gesture to text using the translation model
                inputs = tokenizer(str(gesture_index), return_tensors="pt", padding=True)
                outputs = translation_model.generate(**inputs)
                interpretation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                result["gesture"] = gesture_index
                result["interpretation"] = interpretation
                result["emoji"] = "👋"  # You can map different gestures to different emojis
        
        # Process emotions if face detected
        if face_results.multi_face_landmarks:
            # Use FER for emotion detection
            emotions = emotion_detector.detect_emotions(frame)
            if emotions:
                # Get the dominant emotion
                dominant_emotion = max(emotions[0]['emotions'].items(), key=lambda x: x[1])
                result["emotion"] = dominant_emotion[0]
                result["emotion_confidence"] = float(dominant_emotion[1])
                
                # Map emotions to emojis
                emotion_emojis = {
                    "angry": "😠",
                    "disgust": "🤢",
                    "fear": "😨",
                    "happy": "😊",
                    "sad": "😢",
                    "surprise": "😲",
                    "neutral": "😐"
                }
                result["emoji"] = emotion_emojis.get(result["emotion"], "😐")
    else:
        # If MediaPipe is not available, just return the basic result
        result["interpretation"] = "MediaPipe not available - gesture recognition disabled"
    
    return result

@app.route('/api/recognize', methods=['POST'])
def recognize_gesture():
    try:
        # Get the image data and language from the request
        data = request.json
        image_data = data['frame'].split(',')[1]
        language = data.get('language', 'asl')  # Default to ASL if not specified
        
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process the frame with the specified language
        result = process_frame(frame, language)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WebRTC signaling endpoints
@app.route('/api/call/create', methods=['POST'])
def create_call():
    data = request.json
    username = data.get('username')
    
    if not username or username not in users:
        return jsonify({'error': 'Invalid username'}), 400
    
    call_id = str(uuid.uuid4())
    active_calls[call_id] = {
        'offer': None,
        'answer': None,
        'candidates': [],
        'created_at': datetime.datetime.now().isoformat(),
        'created_by': username,
        'status': 'waiting'
    }
    
    return jsonify({'call_id': call_id}), 201

@app.route('/api/call/<call_id>/offer', methods=['POST'])
def set_offer(call_id):
    try:
        if call_id not in active_calls:
            return jsonify({"error": "Call not found"}), 404
        
        data = request.json
        active_calls[call_id]['offer'] = data['offer']
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/call/<call_id>/answer', methods=['POST'])
def set_answer(call_id):
    try:
        if call_id not in active_calls:
            return jsonify({"error": "Call not found"}), 404
        
        data = request.json
        active_calls[call_id]['answer'] = data['answer']
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/call/<call_id>/candidate', methods=['POST'])
def add_candidate(call_id):
    try:
        if call_id not in active_calls:
            return jsonify({"error": "Call not found"}), 404
        
        data = request.json
        active_calls[call_id]['candidates'].append(data['candidate'])
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/call/<call_id>/get', methods=['GET'])
def get_call_data(call_id):
    if call_id not in active_calls:
        return jsonify({'error': 'Call not found'}), 404
    
    call_data = active_calls[call_id]
    
    # Update call status if needed
    if call_data['status'] == 'waiting' and call_data['answer']:
        call_data['status'] = 'connected'
    
    return jsonify(call_data), 200

@app.route('/api/call/<call_id>/end', methods=['POST'])
def end_call(call_id):
    try:
        if call_id not in active_calls:
            return jsonify({"error": "Call not found"}), 404
        
        del active_calls[call_id]
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def text_to_sign(text, language='asl'):
    """
    Convert text to sign language gestures with support for phrases and sequences
    """
    # First, try to match complete phrases
    for phrase in COMMON_PHRASES:
        if phrase in text.lower():
            sequence = get_phrase_sequence(phrase, language)
            if sequence:
                return {
                    'type': 'sequence',
                    'signs': sequence,
                    'original_text': text
                }
    
    # If no phrase match, process individual words
    sentences = sent_tokenize(text)
    result = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            # Get wordnet synsets for better word understanding
            synsets = wordnet.synsets(word.lower())
            if synsets:
                # Use the most common meaning
                lemma = synsets[0].lemmas()[0].name()
                gesture = get_gesture_mapping(lemma, language)
                if gesture:
                    result.append(gesture)
    
    return {
        'type': 'words',
        'signs': result,
        'original_text': text
    }

@app.route('/api/voice_to_sign', methods=['POST'])
def voice_to_sign_endpoint():
    try:
        data = request.json
        text = data.get('text', '')
        language = data.get('language', 'asl')
        
        # Convert text to sign language gestures
        result = text_to_sign(text, language)
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/')
def index():
    return render_template('obs_test.html')

def process_frames():
    global current_frame, is_processing
    
    while is_processing:
        with frame_lock:
            if obs_camera is not None and obs_camera.isOpened():
                ret, frame = obs_camera.read()
                if ret:
                    # Process frame with MediaPipe if available
                    if MEDIAPIPE_AVAILABLE:
                        mp_hands = mp.solutions.hands
                        with mp_hands.Hands(
                            static_image_mode=False,
                            max_num_hands=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5
                        ) as hands:
                            # Convert BGR to RGB
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = hands.process(rgb_frame)
                            
                            # Draw hand landmarks
                            if results.multi_hand_landmarks:
                                for hand_landmarks in results.multi_hand_landmarks:
                                    mp.solutions.drawing_utils.draw_landmarks(
                                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    current_frame = frame
        time.sleep(1/30)  # Target 30 FPS

@app.route('/start_obs', methods=['POST'])
def start_obs():
    global obs_camera, is_processing
    
    try:
        # Try to open OBS Virtual Camera (usually index 1 or 2)
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    obs_camera = cap
                    is_processing = True
                    # Start processing thread
                    threading.Thread(target=process_frames, daemon=True).start()
                    return jsonify({'message': 'OBS Virtual Camera started successfully'})
                cap.release()
        
        return jsonify({'error': 'OBS Virtual Camera not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_obs', methods=['POST'])
def stop_obs():
    global obs_camera, is_processing, current_frame
    
    try:
        is_processing = False
        if obs_camera is not None:
            obs_camera.release()
            obs_camera = None
        current_frame = None
        return jsonify({'message': 'OBS Virtual Camera stopped successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if current_frame is not None:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', current_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1/30)  # Target 30 FPS
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Add endpoint to list active calls
@app.route('/api/calls/list', methods=['GET'])
def list_calls():
    username = request.args.get('username')
    
    if not username or username not in users:
        return jsonify({'error': 'Invalid username'}), 400
    
    # Filter calls by user
    user_calls = {}
    for call_id, call_data in active_calls.items():
        if call_data['created_by'] == username:
            user_calls[call_id] = call_data
    
    return jsonify({'calls': user_calls}), 200

# Add endpoint to join a call
@app.route('/api/call/<call_id>/join', methods=['POST'])
def join_call(call_id):
    data = request.json
    username = data.get('username')
    
    if not username or username not in users:
        return jsonify({'error': 'Invalid username'}), 400
    
    if call_id not in active_calls:
        return jsonify({'error': 'Call not found'}), 404
    
    call_data = active_calls[call_id]
    
    # Update call status
    call_data['joined_by'] = username
    call_data['status'] = 'connected'
    
    return jsonify({'message': 'Joined call successfully'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
