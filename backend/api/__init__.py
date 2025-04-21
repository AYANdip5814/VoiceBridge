from flask import Flask
from flask_cors import CORS
from backend.config import CORS_ORIGINS

def create_app():
    app = Flask(__name__)
    CORS(app, origins=CORS_ORIGINS)
    
    # Register blueprints
    from .routes import api_bp
    app.register_blueprint(api_bp)
    
    return app 