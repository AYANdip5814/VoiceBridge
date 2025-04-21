import React, { useState } from 'react';
import VideoCall from './components/VideoCall.jsx';
import VoiceRecognition from './components/VoiceRecognition';
import SignOverlay from './components/SignOverlay';
import MultimodalFeedback from './components/MultimodalFeedback';
import { ThemeProvider, useTheme } from './context/ThemeContext';
import axios from 'axios';
import './App.css';

function AppContent() {
  const [selectedLanguage, setSelectedLanguage] = useState('ASL');
  const [interpretation, setInterpretation] = useState(null);
  const [fontSize, setFontSize] = useState(16);
  const [highContrast, setHighContrast] = useState(false);
  const [showOverlay, setShowOverlay] = useState(false);
  const [currentSign, setCurrentSign] = useState(null);
  const { isDarkMode, toggleDarkMode } = useTheme();

  const handleInterpretationUpdate = (data) => {
    setInterpretation(data);
  };

  const handleSpeechRecognized = async (text) => {
    try {
      const response = await axios.post('/api/voice_to_sign', {
        text,
        language: selectedLanguage.toLowerCase()
      });

      if (response.data.success && response.data.signs.length > 0) {
        const sign = response.data.signs[0];
        setCurrentSign(sign);
        setShowOverlay(true);
        
        // Hide overlay after 3 seconds
        setTimeout(() => {
          setShowOverlay(false);
          setCurrentSign(null);
        }, 3000);
      }
    } catch (error) {
      console.error('Error converting speech to sign:', error);
    }
  };

  const handleFeedbackComplete = () => {
    // Additional actions after feedback is complete
  };

  const baseClasses = `
    min-h-screen transition-colors duration-200
    ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-900'}
    ${highContrast ? 'high-contrast' : ''}
  `;

  return (
    <div className={baseClasses} style={{ fontSize: `${fontSize}px` }}>
      <header className={`${isDarkMode ? 'bg-indigo-800' : 'bg-indigo-600'} text-white p-4`}>
        <div className="container mx-auto flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold">VoiceBridge</h1>
            <p className="text-indigo-200">Real-time Sign Language Interpretation</p>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-lg bg-opacity-20 hover:bg-opacity-30 transition-all"
              aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {isDarkMode ? '🌞' : '🌙'}
            </button>
            <button
              onClick={() => setHighContrast(!highContrast)}
              className="p-2 rounded-lg bg-opacity-20 hover:bg-opacity-30 transition-all"
              aria-label={highContrast ? 'Disable high contrast' : 'Enable high contrast'}
            >
              {highContrast ? '👁️' : '👁️‍🗨️'}
            </button>
            <div className="flex items-center space-x-2">
              <label htmlFor="fontSize" className="text-sm">Text Size:</label>
              <input
                type="range"
                id="fontSize"
                min="12"
                max="24"
                value={fontSize}
                onChange={(e) => setFontSize(Number(e.target.value))}
                className="w-24"
                aria-label="Adjust text size"
              />
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto p-4">
        <div className="mb-6">
          <label className="block text-sm font-bold mb-2">
            Select Sign Language:
          </label>
          <select
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
            className={`shadow border rounded py-2 px-3 leading-tight focus:outline-none focus:shadow-outline
              ${isDarkMode ? 'bg-gray-800 text-white border-gray-600' : 'bg-white text-gray-700 border-gray-300'}`}
            aria-label="Select sign language"
          >
            <option value="ASL">American Sign Language (ASL)</option>
            <option value="BSL">British Sign Language (BSL)</option>
            <option value="ISL">Indian Sign Language (ISL)</option>
          </select>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className={`rounded-lg shadow-lg p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <h2 className="text-xl font-bold mb-4">Video Call</h2>
            <VideoCall
              selectedLanguage={selectedLanguage}
              onInterpretationUpdate={handleInterpretationUpdate}
            />
          </div>

          <div className={`rounded-lg shadow-lg p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <h2 className="text-xl font-bold mb-4">Voice to Sign</h2>
            <VoiceRecognition onSpeechRecognized={handleSpeechRecognized} />
          </div>

          <div className={`rounded-lg shadow-lg p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <h2 className="text-xl font-bold mb-4">Interpretation</h2>
            {interpretation ? (
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <span className="text-2xl" role="img" aria-label="Gesture emoji">
                    {interpretation.emoji}
                  </span>
                  <span className="text-lg font-semibold">{interpretation.gesture}</span>
                </div>
                <p>{interpretation.interpretation}</p>
                {interpretation.emotion && (
                  <div className="mt-4">
                    <p className="text-sm opacity-75">Detected Emotion:</p>
                    <p className="text-lg font-medium">{interpretation.emotion}</p>
                  </div>
                )}
              </div>
            ) : (
              <p className="opacity-75">No interpretation available yet.</p>
            )}
          </div>
        </div>
      </main>

      {interpretation && (
        <MultimodalFeedback
          gesture={interpretation.gesture}
          interpretation={interpretation.interpretation}
          emotion={interpretation.emotion}
          landmarks={interpretation.landmarks}
          onFeedbackComplete={handleFeedbackComplete}
        />
      )}

      {currentSign && (
        <SignOverlay
          gesture={currentSign.gesture}
          interpretation={currentSign.word}
          landmarks={currentSign.landmarks}
          isVisible={showOverlay}
        />
      )}
    </div>
  );
}

function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}

export default App;
