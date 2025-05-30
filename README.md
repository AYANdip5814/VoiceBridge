# VoiceBridge

VoiceBridge is a real-time sign language interpreter for video calls, powered by computer vision and AI. It enables seamless communication between hearing and deaf/hard-of-hearing individuals by providing real-time translation of Indian Sign Language (ISL).

## Features

- Real-time video capture and processing
- Indian Sign Language (ISL) recognition
- Text-to-speech conversion
- Speech-to-text conversion
- User-friendly interface
- Support for multiple video sources
- Pause/Resume functionality
- Settings customization

## Tech Stack

### Frontend
- React with TypeScript
- Chakra UI for components
- Electron for desktop application
- WebRTC for video handling

### Backend
- FastAPI
- TensorFlow for ML models
- OpenCV for video processing
- MediaPipe for pose estimation
- Python for backend services

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VoiceBridge.git
cd VoiceBridge
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
```

3. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

## Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

3. For the desktop application:
```bash
cd frontend
npm run electron-dev
```

## Project Structure

```
VoiceBridge/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── types/
│   ├── public/
│   └── package.json
├── backend/
│   ├── ml/
│   │   ├── model.py
│   │   └── utils.py
│   ├── main.py
│   └── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for pose estimation
- TensorFlow team for the ML framework
- The open-source community for various tools and libraries

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/VoiceBridge #   V o i c e B r i d g e  
 #   V o i c e B r i d g e  
 