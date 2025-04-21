# backend/gesture_recognition.py

def recognize_gesture(landmarks):
    # Dummy gesture recognition based on the number of landmarks
    if not landmarks:
        return "No hand detected"
    
    # Example: basic check on y-coordinates of landmarks
    finger_tip = landmarks[8]  # Index finger tip
    thumb_tip = landmarks[4]  # Thumb tip

    if abs(finger_tip.y - thumb_tip.y) < 0.05:
        return "Hello"
    else:
        return "Yes"
