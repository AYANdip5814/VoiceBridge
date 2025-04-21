// src/App.jsx

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Auth from './components/Auth';
import VideoCall from './components/VideoCall';
import VoiceRecognition from './components/VoiceRecognition';
import SignOverlay from './components/SignOverlay';
import MultimodalFeedback from './components/MultimodalFeedback';
import Profile from './components/Profile';
import Navigation from './components/Navigation';
import './App.css';

// Protected Route component
const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem('token');
  if (!token) {
    return <Navigate to="/auth" replace />;
  }
  return children;
};

// Main App Content component
const AppContent = () => {
  return (
    <div className="app-container">
      <Navigation />
      <Routes>
        <Route path="/auth" element={<Auth />} />
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <div className="min-h-screen">
                <header>
                  <div className="container">
                    <h1>VoiceBridge</h1>
                    <p className="text-light">Breaking communication barriers</p>
                  </div>
                </header>
                <main>
                  <div className="container">
                    <div className="grid">
                      <div className="card">
                        <h2>Video Call</h2>
                        <p className="text-muted mb-4">Start a video call with sign language translation</p>
                        <VideoCall />
                      </div>
                      <div className="card">
                        <h2>Voice Recognition</h2>
                        <p className="text-muted mb-4">Convert speech to text with real-time translation</p>
                        <VoiceRecognition />
                      </div>
                    </div>
                    <div className="grid mt-4">
                      <div className="card">
                        <h2>Sign Language Overlay</h2>
                        <p className="text-muted mb-4">Real-time sign language detection and translation</p>
                        <SignOverlay />
                      </div>
                      <div className="card">
                        <h2>Multimodal Feedback</h2>
                        <p className="text-muted mb-4">Enhanced communication with visual and audio feedback</p>
                        <MultimodalFeedback />
                      </div>
                    </div>
                  </div>
                </main>
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/profile"
          element={
            <ProtectedRoute>
              <div className="min-h-screen">
                <header>
                  <div className="container">
                    <h1>Profile</h1>
                  </div>
                </header>
                <main>
                  <div className="container">
                    <div className="card">
                      <Profile />
                    </div>
                  </div>
                </main>
              </div>
            </ProtectedRoute>
          }
        />
      </Routes>
    </div>
  );
};

// Main App component
function App() {
  return (
    <ThemeProvider>
      <Router>
        <AppContent />
      </Router>
    </ThemeProvider>
  );
}

export default App;
