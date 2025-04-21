import React, { useState, useEffect, useRef } from 'react';
import { useTheme } from '../context/ThemeContext';

const VoiceRecognition = ({ onSpeechRecognized, isProcessing }) => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [error, setError] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const recognitionRef = useRef(null);
  const { isDarkMode } = useTheme();

  useEffect(() => {
    // Check if browser supports speech recognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event) => {
        const current = event.resultIndex;
        const result = event.results[current];
        const transcriptText = result[0].transcript;
        const currentConfidence = result[0].confidence;
        
        setTranscript(transcriptText);
        setConfidence(currentConfidence);
        
        if (result.isFinal) {
          onSpeechRecognized(transcriptText);
        }
      };

      recognitionRef.current.onerror = (event) => {
        setError(`Error: ${event.error}`);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    } else {
      setError('Speech recognition is not supported in this browser.');
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [onSpeechRecognized]);

  const toggleListening = () => {
    if (!recognitionRef.current) return;

    if (isListening) {
      recognitionRef.current.stop();
    } else {
      setTranscript('');
      setError(null);
      setConfidence(0);
      recognitionRef.current.start();
      setIsListening(true);
    }
  };

  return (
    <div className={`p-4 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-lg`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Voice Recognition</h3>
        <div className="flex items-center space-x-2">
          {isProcessing && (
            <span className="text-sm opacity-75">Converting to signs...</span>
          )}
          <button
            onClick={toggleListening}
            disabled={isProcessing}
            className={`px-4 py-2 rounded-full transition-colors ${
              isListening
                ? 'bg-red-500 hover:bg-red-600'
                : 'bg-green-500 hover:bg-green-600'
            } text-white ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
            aria-label={isListening ? 'Stop listening' : 'Start listening'}
          >
            {isListening ? 'Stop' : 'Start'} Listening
          </button>
        </div>
      </div>

      {error && (
        <div className="text-red-500 mb-4">
          {error}
        </div>
      )}

      <div className="relative">
        <div className={`p-4 rounded-lg ${
          isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
        } min-h-[100px]`}>
          {transcript ? (
            <div className="space-y-2">
              <p>{transcript}</p>
              {confidence > 0 && (
                <div className="flex items-center space-x-2">
                  <div className="flex-1 h-1 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-green-500 transition-all duration-300"
                      style={{ width: `${confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs opacity-75">
                    {Math.round(confidence * 100)}% confidence
                  </span>
                </div>
              )}
            </div>
          ) : (
            <p className="opacity-50">
              {isListening ? 'Listening...' : 'Click "Start Listening" to begin'}
            </p>
          )}
        </div>
        {isListening && (
          <div className="absolute top-2 right-2">
            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
          </div>
        )}
      </div>

      <div className="mt-4 text-sm opacity-75">
        <p>Try saying common phrases like:</p>
        <ul className="list-disc list-inside mt-1">
          <li>"Hello, how are you?"</li>
          <li>"Thank you very much"</li>
          <li>"Please help me"</li>
          <li>"Nice to meet you"</li>
        </ul>
      </div>
    </div>
  );
};

export default VoiceRecognition; 