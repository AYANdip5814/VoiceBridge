# backend/utils/hand_tracking.py

import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

def get_hand_landmarks(image):
    results = hands.process(image)
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0].landmark
    return None
