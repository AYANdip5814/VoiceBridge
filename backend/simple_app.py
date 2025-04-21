from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)

# Global variables for OBS camera
obs_camera = None
frame_lock = threading.Lock()
current_frame = None
is_processing = False

def process_frames():
    global current_frame, is_processing
    
    while is_processing:
        with frame_lock:
            if obs_camera is not None and obs_camera.isOpened():
                ret, frame = obs_camera.read()
                if ret:
                    # Just store the frame without any processing
                    current_frame = frame
        time.sleep(1/30)  # Target 30 FPS

@app.route('/')
def index():
    return render_template('obs_test.html')

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 