import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const VideoCall = ({ selectedLanguage }) => {
  const [callStatus, setCallStatus] = useState('idle');
  const [callId, setCallId] = useState(null);
  const [error, setError] = useState(null);
  const [showInterpretation, setShowInterpretation] = useState(true);
  const [remoteGesture, setRemoteGesture] = useState(null);

  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const localStreamRef = useRef(null);
  const interpretationIntervalRef = useRef(null);
  const pollingIntervalRef = useRef(null);

  const ICE_SERVERS = {
    iceServers: [
      { urls: 'stun:stun.l.google.com:19302' },
      { urls: 'stun:stun1.l.google.com:19302' },
      { urls: 'stun:stun2.l.google.com:19302' },
    ],
  };

  const createPeerConnection = () => {
    try {
      const pc = new RTCPeerConnection(ICE_SERVERS);

      pc.onicecandidate = async (event) => {
        if (event.candidate && callId) {
          try {
            await axios.post(`http://127.0.0.1:5000/api/call/${callId}/candidate`, {
              candidate: event.candidate,
            });
          } catch (error) {
            console.error('Error sending ICE candidate:', error);
          }
        }
      };

      pc.oniceconnectionstatechange = () => {
        if (pc.iceConnectionState === 'disconnected') {
          handleDisconnect();
        }
      };

      pc.ontrack = (event) => {
        if (remoteVideoRef.current && event.streams[0]) {
          remoteVideoRef.current.srcObject = event.streams[0];
        }
      };

      return pc;
    } catch (error) {
      console.error('Error creating peer connection:', error);
      setError('Failed to create peer connection');
      return null;
    }
  };

  const handleDisconnect = async () => {
    setCallStatus('idle');
    setError('Call disconnected');
    await endCall();
  };

  const startCall = async () => {
    try {
      setCallStatus('connecting');
      setError(null);

      // Create a new call session
      const response = await axios.post('http://127.0.0.1:5000/api/call/create');
      const newCallId = response.data.call_id;
      setCallId(newCallId);

      // Get local media stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      
      localStreamRef.current = stream;
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
      }

      // Create and configure peer connection
      const pc = createPeerConnection();
      if (!pc) return;

      peerConnectionRef.current = pc;

      stream.getTracks().forEach((track) => {
        pc.addTrack(track, stream);
      });

      // Create and set local description
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      
      await axios.post(`http://127.0.0.1:5000/api/call/${newCallId}/offer`, {
        offer: offer,
      });

      // Start polling for answer
      startPolling(newCallId);
      setCallStatus('connected');
    } catch (error) {
      console.error('Error starting call:', error);
      setError(error.message || 'Failed to start call');
      setCallStatus('idle');
    }
  };

  const startPolling = (id) => {
    pollingIntervalRef.current = setInterval(async () => {
      try {
        const response = await axios.get(`http://127.0.0.1:5000/api/call/${id}/get`);
        const callData = response.data;

        if (callData.answer && peerConnectionRef.current && !peerConnectionRef.current.currentRemoteDescription) {
          await peerConnectionRef.current.setRemoteDescription(
            new RTCSessionDescription(callData.answer)
          );

          // Add any pending ICE candidates
          if (callData.candidates) {
            for (const candidate of callData.candidates) {
              try {
                await peerConnectionRef.current.addIceCandidate(
                  new RTCIceCandidate(candidate)
                );
              } catch (error) {
                console.error('Error adding ICE candidate:', error);
              }
            }
          }
        }
      } catch (error) {
        console.error('Error polling for answer:', error);
      }
    }, 1000);
  };

  const endCall = async () => {
    try {
      if (callId) {
        await axios.post(`http://127.0.0.1:5000/api/call/${callId}/end`);
      }

      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach((track) => track.stop());
        localStreamRef.current = null;
      }

      if (peerConnectionRef.current) {
        peerConnectionRef.current.close();
        peerConnectionRef.current = null;
      }

      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }

      if (interpretationIntervalRef.current) {
        clearInterval(interpretationIntervalRef.current);
        interpretationIntervalRef.current = null;
      }

      setCallStatus('idle');
      setCallId(null);
      setRemoteGesture(null);
    } catch (error) {
      console.error('Error ending call:', error);
      setError('Failed to end call properly');
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      endCall();
    };
  }, []);

  // Start interpretation when call is connected
  useEffect(() => {
    if (callStatus === 'connected' && showInterpretation) {
      interpretationIntervalRef.current = setInterval(async () => {
        try {
          const canvas = document.createElement('canvas');
          const video = remoteVideoRef.current;
          
          if (!video || !video.videoWidth) return;
          
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(video, 0, 0);
          
          const frame = canvas.toDataURL('image/jpeg');
          const response = await axios.post('http://127.0.0.1:5000/api/recognize', {
            frame,
            language: selectedLanguage,
          });

          setRemoteGesture(response.data);
        } catch (error) {
          console.error('Error processing remote frame:', error);
        }
      }, 1000);
    }

    return () => {
      if (interpretationIntervalRef.current) {
        clearInterval(interpretationIntervalRef.current);
      }
    };
  }, [callStatus, showInterpretation, selectedLanguage]);

  return (
    <div className="flex flex-col items-center space-y-4 p-4 w-full max-w-6xl">
      <div className="grid grid-cols-2 gap-4 w-full">
        <div className="relative">
          <video
            ref={localVideoRef}
            autoPlay
            playsInline
            muted
            className="w-full rounded-lg bg-gray-800"
          />
          <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded">
            You
          </div>
        </div>
        <div className="relative">
          <video
            ref={remoteVideoRef}
            autoPlay
            playsInline
            className="w-full rounded-lg bg-gray-800"
          />
          <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded">
            Remote
          </div>
          {remoteGesture && (
            <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded">
              {remoteGesture.interpretation} {remoteGesture.emoji}
            </div>
          )}
        </div>
      </div>

      <div className="flex space-x-4">
        {callStatus === 'idle' ? (
          <button
            onClick={startCall}
            className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 transition-colors"
          >
            Start Call
          </button>
        ) : (
          <button
            onClick={endCall}
            className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 transition-colors"
          >
            End Call
          </button>
        )}
        <button
          onClick={() => setShowInterpretation(!showInterpretation)}
          className={`px-4 py-2 rounded transition-colors ${
            showInterpretation
              ? 'bg-blue-500 text-white hover:bg-blue-600'
              : 'bg-gray-500 text-white hover:bg-gray-600'
          }`}
        >
          {showInterpretation ? 'Hide Interpretation' : 'Show Interpretation'}
        </button>
      </div>

      {error && (
        <div className="text-red-500 text-center bg-red-100 p-2 rounded w-full">
          {error}
        </div>
      )}

      {callStatus === 'connecting' && (
        <div className="text-yellow-500 text-center">
          Connecting to call...
        </div>
      )}
    </div>
  );
};

export default VideoCall; 