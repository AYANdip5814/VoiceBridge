import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import styled from "styled-components";  // Import styled-components
import './WebCapture.css';  // Import the external CSS file

// Video capture configuration
const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
};

// Create the styled component for the overlay
const RecognitionOverlay = styled.div`
  position: absolute;
  bottom: 20px;
  left: 20px;
  background-color: rgba(255, 255, 255, 0.7);
  padding: 10px;
  border-radius: 10px;
  font-family: "Arial", sans-serif;
  font-size: 16px;
`;

const WebcamCapture = () => {
    const webcamRef = useRef(null);  // Ref for the webcam feed
    const [response, setResponse] = useState(null);  // To hold recognized gesture and interpretation
    const [isProcessing, setIsProcessing] = useState(false);  // Flag to prevent overlapping requests

    // Function to capture frame and send to backend
    const captureFrame = async () => {
        if (isProcessing) return; // Prevent multiple requests at once

        const imageSrc = webcamRef.current.getScreenshot(); // Capture current frame

        // Send image to backend for gesture recognition
        try {
            setIsProcessing(true);
            const res = await axios.post("http://localhost:5000/api/recognize", {
                image: imageSrc,
            });
            setResponse(res.data); // Set the backend response with gesture & interpretation
        } catch (err) {
            console.error("Error sending frame to backend:", err);
        } finally {
            setIsProcessing(false); // Allow next frame processing
        }
    };

    // Set an interval to capture frames every 1000ms (1 second)
    useEffect(() => {
        const interval = setInterval(() => {
            captureFrame(); // Capture a new frame at intervals
        }, 1000); // Adjust this value for performance

        return () => clearInterval(interval); // Cleanup on unmount
    }, [isProcessing]);

    return (
        <div className="flex flex-col items-center gap-4">
            <div className="relative">
                {/* Live Webcam Feed */}
                <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                    className="rounded-xl shadow-md"
                />

                {/* Display recognized gesture using the styled component */}
                {response && !isProcessing && (
                    <RecognitionOverlay>
                        <p className="font-bold">Gesture: {response?.gesture}</p>
                        <p>Interpretation: {response?.interpretation}</p>
                    </RecognitionOverlay>
                )}
            </div>

            {/* Processing state */}
            {isProcessing && (
                <div className="mt-4 text-center">
                    <p className="text-lg font-bold">Processing...</p>
                </div>
            )}
        </div>
    );
};

export default WebcamCapture;
