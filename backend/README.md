# VoiceBridge Backend

This directory contains the Python backend for VoiceBridge, handling video processing and sign language translation.

## Project Structure

```
backend/
├── api/              # Flask API endpoints
├── ml/              # Machine learning models and inference
├── utils/           # Utility functions and helpers
├── tests/           # Unit tests
└── config.py        # Configuration settings
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
python run.py
```

## API Endpoints

- `/api/v1/translate` - Real-time sign language translation
- `/api/v1/sources` - Get available video sources
- `/api/v1/settings` - Get/Update application settings

## Development

- Run tests: `pytest`
- Format code: `black .`
- Lint code: `flake8` 