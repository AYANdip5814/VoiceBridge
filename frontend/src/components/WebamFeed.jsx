import React, { useEffect, useRef, useState } from 'react';
import Webcam from "react-webcam";
import axios from "axios";
import ping from "../assets/ping.mp3";

const WebcamFeed = () => {
    const webcamRef = useRef(null);
    const [gesture, setGesture] = useState("");
    const [emoji, setEmoji] = useState("");
    const [history, setHistory] = useState([]);
    const audio = new Audio(ping);

    const capture = async () => {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
            try {
                const response = await axios.post("http://127.0.0.1:5000/api/recognize", {
                    frame: imageSrc
                });

                const { gesture, emoji, interpretation } = response.data;
                setGesture(interpretation);
                setEmoji(emoji);
                setHistory(prev => [...prev.slice(-9), interpretation]);
                audio.play();
            } catch (error) {
                console.error("Error sending frame:", error);
            }
        }
    };

    useEffect(() => {
        const interval = setInterval(capture, 130); // ~0.013 sec
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="flex flex-col items-center p-4 bg-gradient-to-tr from-indigo-800 to-purple-900 text-white min-h-screen">
            <h1 className="text-4xl font-bold mb-4">VoiceBridge Live Interpreter</h1>
            <Webcam
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                width={480}
                height={360}
                className="rounded-lg border-4 border-purple-600"
            />
            <p className="mt-4 text-2xl">{gesture}</p>
            <div className="mt-4 w-full max-w-md bg-indigo-700 p-4 rounded">
                <h2 className="text-lg font-semibold mb-2">Gesture History</h2>
                <ul className="space-y-1 text-purple-100 text-sm max-h-40 overflow-y-auto">
                    {history.slice().reverse().map((item, index) => (
                        <li key={index} className="border-b border-purple-400 pb-1">{item}</li>
                    ))}
                </ul>
            </div>
        </div>
    );
};

export default WebcamFeed;
