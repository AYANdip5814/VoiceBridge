from flask import Blueprint, jsonify, request
from backend.ml.translator import SignLanguageTranslator
from backend.utils.video import get_video_sources

api_bp = Blueprint('api', __name__)
translator = SignLanguageTranslator()

@api_bp.route('/sources', methods=['GET'])
def get_sources():
    """Get available video sources."""
    try:
        sources = get_video_sources()
        return jsonify({'sources': sources})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/translate', methods=['POST'])
def translate():
    """Translate sign language from video frame."""
    try:
        frame_data = request.get_json()
        if not frame_data or 'frame' not in frame_data:
            return jsonify({'error': 'No frame data provided'}), 400
            
        result = translator.translate_frame(frame_data['frame'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/settings', methods=['GET', 'POST'])
def settings():
    """Get or update application settings."""
    if request.method == 'GET':
        return jsonify({
            'confidence_threshold': translator.confidence_threshold,
            'model_path': translator.model_path
        })
    else:
        data = request.get_json()
        if 'confidence_threshold' in data:
            translator.confidence_threshold = float(data['confidence_threshold'])
        return jsonify({'message': 'Settings updated successfully'}) 