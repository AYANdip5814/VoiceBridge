"""
Comprehensive gesture mappings for different sign languages
"""
from math import cos, sin, pi

# Extended hand positions/shapes used across different signs
HAND_SHAPES = {
    'fist': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    'open_palm': [0.5, 0.5, 0.7, 0.5, 0.9, 0.5],
    'point': [0.5, 0.5, 0.7, 0.3, 0.9, 0.1],
    'pinch': [0.5, 0.5, 0.6, 0.4, 0.7, 0.3],
    'peace': [0.5, 0.5, 0.7, 0.3, 0.7, 0.3],
    'thumbs_up': [0.5, 0.5, 0.5, 0.3, 0.5, 0.2],
    'namaste': [0.5, 0.5, 0.5, 0.4, 0.5, 0.3],
    'flat_hand': [0.5, 0.5, 0.7, 0.5, 0.9, 0.5],
    'claw': [0.5, 0.5, 0.6, 0.4, 0.7, 0.3],
    'ok_sign': [0.5, 0.5, 0.6, 0.4, 0.7, 0.3],
}

# Extended movement patterns
MOVEMENTS = {
    'up': lambda x, y: (x, y - 0.1),
    'down': lambda x, y: (x, y + 0.1),
    'left': lambda x, y: (x - 0.1, y),
    'right': lambda x, y: (x + 0.1, y),
    'circle': lambda x, y, t: (
        x + 0.1 * cos(t * 2 * pi),
        y + 0.1 * sin(t * 2 * pi)
    ),
    'wave': lambda x, y, t: (
        x + 0.05 * sin(t * 4 * pi),
        y + 0.05 * cos(t * 2 * pi)
    ),
    'spiral': lambda x, y, t: (
        x + t * 0.1 * cos(t * 4 * pi),
        y + t * 0.1 * sin(t * 4 * pi)
    ),
    'zigzag': lambda x, y, t: (
        x + t * 0.2,
        y + 0.1 * sin(t * 8 * pi)
    ),
    'bounce': lambda x, y, t: (
        x,
        y + 0.1 * abs(sin(t * 2 * pi))
    ),
}

# Extended ASL Gesture Mappings
ASL_GESTURES = {
    # Greetings and Common Phrases
    'hello': {
        'hand_shape': 'open_palm',
        'movement': ['up', 'wave'],
        'landmarks': HAND_SHAPES['open_palm'],
        'duration': 1.0
    },
    'goodbye': {
        'hand_shape': 'open_palm',
        'movement': ['wave'],
        'landmarks': HAND_SHAPES['open_palm'],
        'duration': 1.2
    },
    'thank_you': {
        'hand_shape': 'flat_hand',
        'movement': ['forward', 'down'],
        'landmarks': HAND_SHAPES['flat_hand'],
        'duration': 0.8
    },
    'please': {
        'hand_shape': 'flat_hand',
        'movement': ['circle'],
        'landmarks': HAND_SHAPES['flat_hand'],
        'duration': 1.2
    },
    
    # Basic Responses
    'yes': {
        'hand_shape': 'fist',
        'movement': ['nod'],
        'landmarks': HAND_SHAPES['fist'],
        'duration': 0.5
    },
    'no': {
        'hand_shape': 'point',
        'movement': ['shake'],
        'landmarks': HAND_SHAPES['point'],
        'duration': 0.5
    },
    
    # Actions
    'help': {
        'hand_shape': 'fist',
        'movement': ['up'],
        'landmarks': HAND_SHAPES['fist'],
        'duration': 1.0
    },
    'want': {
        'hand_shape': 'pinch',
        'movement': ['pull'],
        'landmarks': HAND_SHAPES['pinch'],
        'duration': 0.8
    },
    'eat': {
        'hand_shape': 'pinch',
        'movement': ['mouth'],
        'landmarks': HAND_SHAPES['pinch'],
        'duration': 0.8
    },
    'drink': {
        'hand_shape': 'claw',
        'movement': ['mouth'],
        'landmarks': HAND_SHAPES['claw'],
        'duration': 0.8
    },
    
    # Emotions
    'happy': {
        'hand_shape': 'flat_hand',
        'movement': ['circle'],
        'landmarks': HAND_SHAPES['flat_hand'],
        'duration': 1.0
    },
    'sad': {
        'hand_shape': 'flat_hand',
        'movement': ['down'],
        'landmarks': HAND_SHAPES['flat_hand'],
        'duration': 1.2
    },
    'love': {
        'hand_shape': 'fist',
        'movement': ['cross_heart'],
        'landmarks': HAND_SHAPES['fist'],
        'duration': 1.0
    },
    
    # Questions
    'what': {
        'hand_shape': 'open_palm',
        'movement': ['zigzag'],
        'landmarks': HAND_SHAPES['open_palm'],
        'duration': 0.8
    },
    'where': {
        'hand_shape': 'point',
        'movement': ['circle'],
        'landmarks': HAND_SHAPES['point'],
        'duration': 0.8
    },
    'when': {
        'hand_shape': 'point',
        'movement': ['circle'],
        'landmarks': HAND_SHAPES['point'],
        'duration': 0.8
    },
    'who': {
        'hand_shape': 'point',
        'movement': ['zigzag'],
        'landmarks': HAND_SHAPES['point'],
        'duration': 0.8
    },
}

# Extended BSL Gesture Mappings
BSL_GESTURES = {
    'hello': {
        'hand_shape': 'wave',
        'movement': ['wave'],
        'landmarks': HAND_SHAPES['open_palm'],
        'duration': 1.0
    },
    'goodbye': {
        'hand_shape': 'wave',
        'movement': ['wave'],
        'landmarks': HAND_SHAPES['open_palm'],
        'duration': 1.2
    },
    'please': {
        'hand_shape': 'flat_hand',
        'movement': ['circle'],
        'landmarks': HAND_SHAPES['flat_hand'],
        'duration': 0.8
    },
    'thank_you': {
        'hand_shape': 'flat_hand',
        'movement': ['chin_forward'],
        'landmarks': HAND_SHAPES['flat_hand'],
        'duration': 0.8
    },
}

# Extended ISL Gesture Mappings
ISL_GESTURES = {
    'hello': {
        'hand_shape': 'namaste',
        'movement': ['join'],
        'landmarks': HAND_SHAPES['namaste'],
        'duration': 1.0
    },
    'namaste': {
        'hand_shape': 'namaste',
        'movement': ['join'],
        'landmarks': HAND_SHAPES['namaste'],
        'duration': 1.2
    },
    'thank_you': {
        'hand_shape': 'flat_hand',
        'movement': ['heart_out'],
        'landmarks': HAND_SHAPES['flat_hand'],
        'duration': 1.0
    },
}

# Extended Common Phrases
COMMON_PHRASES = {
    'how are you': ['how', 'you'],
    'thank you very much': ['thank_you', 'very', 'much'],
    'please help me': ['please', 'help', 'me'],
    'nice to meet you': ['nice', 'meet', 'you'],
    'what is your name': ['what', 'your', 'name'],
    'i love you': ['i', 'love', 'you'],
    'good morning': ['good', 'morning'],
    'good night': ['good', 'night'],
    'excuse me': ['excuse', 'me'],
    'i understand': ['i', 'understand'],
    'i dont understand': ['i', 'no', 'understand'],
}

# Emotion mappings to gestures
EMOTION_GESTURES = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'surprised': 'surprised',
    'confused': 'confused',
    'excited': 'excited',
}

def get_gesture_mapping(word, language='asl'):
    """Get the gesture mapping for a word in the specified sign language"""
    language = language.lower()
    if language == 'asl':
        return ASL_GESTURES.get(word, None)
    elif language == 'bsl':
        return BSL_GESTURES.get(word, None)
    elif language == 'isl':
        return ISL_GESTURES.get(word, None)
    return None

def get_phrase_sequence(phrase, language='asl'):
    """Get the gesture sequence for a common phrase"""
    normalized_phrase = phrase.lower().strip()
    if normalized_phrase in COMMON_PHRASES:
        return [get_gesture_mapping(word, language) 
                for word in COMMON_PHRASES[normalized_phrase]]
    return None

def get_emotion_gesture(emotion, language='asl'):
    """Get the gesture mapping for an emotion"""
    if emotion in EMOTION_GESTURES:
        return get_gesture_mapping(EMOTION_GESTURES[emotion], language)
    return None 