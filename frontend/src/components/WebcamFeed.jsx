import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import Webcam from "react-webcam";
import axios from "axios";
import ping from "../assets/ping.mp3";
import GestureAnimation from './GestureAnimation';
import Profile from './Profile';
import LanguageSelector from './LanguageSelector';
import VideoCall from './VideoCall';
import Logo from './Logo';

const PERFORMANCE_METRICS_WINDOW = 60; // Keep last 60 frames of metrics
const TARGET_FPS = 77; // Target frame rate (13ms between frames)
const MIN_FPS_THRESHOLD = 45; // Minimum acceptable FPS before quality reduction
const MAX_PROCESSING_TIME = 50; // Maximum allowed processing time in ms

const WebcamFeed = () => {
	const webcamRef = useRef(null);
	const lastCaptureTime = useRef(0);
	const processingRef = useRef(false);
	const [gesture, setGesture] = useState("");
	const [emoji, setEmoji] = useState("");
	const [history, setHistory] = useState([]);
	const [isInterpreting, setIsInterpreting] = useState(true);
	const [volume, setVolume] = useState(50);
	const [isMuted, setIsMuted] = useState(false);
	const [landmarks, setLandmarks] = useState([]);
	const [emotion, setEmotion] = useState("neutral");
	const [emotionConfidence, setEmotionConfidence] = useState(0);
	const [activeTab, setActiveTab] = useState('interpreter');
	const [selectedLanguage, setSelectedLanguage] = useState('asl');
	const [error, setError] = useState(null);
	const [fps, setFps] = useState(0);
	const frameCountRef = useRef(0);
	const lastFpsUpdate = useRef(Date.now());
	const audioRef = useRef(new Audio(ping));

	// Performance monitoring state
	const [performanceMetrics, setPerformanceMetrics] = useState({
		fps: 0,
		avgProcessingTime: 0,
		droppedFrames: 0,
		quality: 1.0
	});
	const metricsHistoryRef = useRef([]);
	const droppedFramesRef = useRef(0);
	const qualityRef = useRef(1.0); // 1.0 = full quality, 0.5 = half resolution, etc.

	// Available sign languages
	const languages = [
		{ code: 'asl', name: 'American Sign Language (ASL)' },
		{ code: 'bsl', name: 'British Sign Language (BSL)' },
		{ code: 'isl', name: 'Irish Sign Language (ISL)' },
		{ code: 'auslan', name: 'Australian Sign Language (Auslan)' }
	];

	const handleLanguageChange = (language) => {
		setSelectedLanguage(language);
		setHistory([]);
		setError(null);
	};

	const updateMetrics = useCallback((processingTime) => {
		const now = Date.now();
		const metrics = {
			timestamp: now,
			processingTime,
			quality: qualityRef.current
		};

		metricsHistoryRef.current.push(metrics);
		if (metricsHistoryRef.current.length > PERFORMANCE_METRICS_WINDOW) {
			metricsHistoryRef.current.shift();
		}

		const elapsed = now - lastFpsUpdate.current;
		if (elapsed >= 1000) {
			const recentMetrics = metricsHistoryRef.current;
			const avgProcessingTime = recentMetrics.reduce((sum, m) => sum + m.processingTime, 0) / recentMetrics.length;
			const currentFps = Math.round((frameCountRef.current * 1000) / elapsed);

			// Adjust quality based on performance
			if (currentFps < MIN_FPS_THRESHOLD && qualityRef.current > 0.25) {
				qualityRef.current = Math.max(0.25, qualityRef.current - 0.25);
			} else if (currentFps > TARGET_FPS && avgProcessingTime < MAX_PROCESSING_TIME && qualityRef.current < 1.0) {
				qualityRef.current = Math.min(1.0, qualityRef.current + 0.25);
			}

			setPerformanceMetrics({
				fps: currentFps,
				avgProcessingTime: Math.round(avgProcessingTime),
				droppedFrames: droppedFramesRef.current,
				quality: qualityRef.current
			});

			frameCountRef.current = 0;
			lastFpsUpdate.current = now;
		}
	}, []);

	// Webcam configuration
	const videoConstraints = useMemo(() => ({
		width: Math.round(640 * qualityRef.current),
		height: Math.round(480 * qualityRef.current),
		facingMode: "user",
		frameRate: TARGET_FPS
	}), []);

	const capture = useCallback(async () => {
		if (!isInterpreting || !webcamRef.current || processingRef.current) {
			droppedFramesRef.current++;
			return;
		}

		const now = performance.now();
		const timeSinceLastCapture = now - lastCaptureTime.current;
		
		if (timeSinceLastCapture < (1000 / TARGET_FPS)) {
			droppedFramesRef.current++;
			return;
		}

		processingRef.current = true;
		lastCaptureTime.current = now;
		frameCountRef.current++;
		
		try {
			const captureStart = performance.now();
			const imageSrc = webcamRef.current.getScreenshot();
			if (!imageSrc) return;

			const response = await axios.post("http://127.0.0.1:5000/api/recognize", {
				frame: imageSrc,
				language: selectedLanguage,
				quality: qualityRef.current
			}, {
				timeout: MAX_PROCESSING_TIME
			});

			const processingTime = performance.now() - captureStart;
			updateMetrics(processingTime);

			const { gesture, emoji, interpretation, landmarks, emotion, emotion_confidence } = response.data;
			
			if (landmarks?.length > 0) {
				setLandmarks(landmarks);
			}
			
			if (emotion) {
				setEmotion(emotion);
				setEmotionConfidence(emotion_confidence);
			}
			
			if (interpretation && interpretation !== "No gesture detected") {
				setGesture(interpretation);
				setEmoji(emoji || "");
				setHistory(prev => [...prev.slice(-9), interpretation]);
				
				if (!isMuted && audioRef.current) {
					audioRef.current.volume = volume / 100;
					audioRef.current.play().catch(console.error);
				}
			}

			setError(null);
		} catch (error) {
			console.error("Error sending frame:", error);
			setError(error.message || "Failed to process frame");
			droppedFramesRef.current++;
		} finally {
			processingRef.current = false;
		}
	}, [isInterpreting, selectedLanguage, isMuted, volume, updateMetrics]);

	useEffect(() => {
		let animationFrameId;
		
		const captureLoop = () => {
			capture();
			animationFrameId = requestAnimationFrame(captureLoop);
		};

		if (isInterpreting) {
			captureLoop();
		}
		
		return () => {
			if (animationFrameId) {
				cancelAnimationFrame(animationFrameId);
			}
		};
	}, [capture, isInterpreting]);

	// Performance metrics display component
	const PerformanceMetrics = () => (
		<div className="flex items-center space-x-4">
			<span className="text-sm font-medium bg-purple-800 px-3 py-1 rounded-full">
				{performanceMetrics.fps} FPS
			</span>
			<span className="text-sm font-medium bg-purple-800 px-3 py-1 rounded-full">
				{performanceMetrics.avgProcessingTime}ms
			</span>
			<span className="text-sm font-medium bg-purple-800 px-3 py-1 rounded-full">
				{Math.round(performanceMetrics.quality * 100)}% Quality
			</span>
			{performanceMetrics.droppedFrames > 0 && (
				<span className="text-sm font-medium bg-red-600 px-3 py-1 rounded-full">
					{performanceMetrics.droppedFrames} Dropped
				</span>
			)}
		</div>
	);

	return (
		<div className="min-h-screen bg-gradient-to-tr from-indigo-800 to-purple-900 text-white">
			{/* Header */}
			<header className="w-full bg-black bg-opacity-20 backdrop-blur-sm shadow-lg">
				<div className="container mx-auto px-4 py-3 flex justify-between items-center">
					<Logo size="medium" />
					<div className="flex items-center space-x-4">
						<PerformanceMetrics />
						<nav className="flex space-x-4">
							{['interpreter', 'video-call', 'profile'].map((tab) => (
								<button
									key={tab}
									onClick={() => setActiveTab(tab)}
									className={`px-4 py-2 rounded-full font-medium transition-all
										${activeTab === tab
											? 'bg-purple-600 text-white shadow-lg'
											: 'text-purple-200 hover:bg-purple-800 hover:text-white'}`}
								>
									{tab.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
								</button>
							))}
						</nav>
					</div>
				</div>
			</header>

			{/* Main Content */}
			<main className="container mx-auto px-4 py-6">
				{error && (
					<div className="mb-6 p-4 bg-red-500 bg-opacity-90 rounded-lg shadow-lg">
						<p className="text-white font-medium">{error}</p>
					</div>
				)}

				{activeTab === 'interpreter' && (
					<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
						{/* Webcam Feed */}
						<div className="lg:col-span-2">
							<div className="relative rounded-lg overflow-hidden shadow-2xl">
								<Webcam
									ref={webcamRef}
									screenshotFormat="image/jpeg"
									className="w-full h-auto"
									onUserMediaError={(err) => setError("Camera access denied: " + err.message)}
								/>
								<GestureAnimation landmarks={landmarks} gesture={gesture} />
								
								{/* Controls Overlay */}
								<div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black to-transparent">
									<div className="flex justify-between items-center">
										<div className="flex space-x-4">
											<button
												onClick={() => setIsInterpreting(!isInterpreting)}
												className={`px-4 py-2 rounded-full font-medium transition-all
													${isInterpreting ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'}`}
											>
												{isInterpreting ? '⏸ Pause' : '▶ Start'}
											</button>
											<button
												onClick={() => setIsMuted(!isMuted)}
												className={`px-4 py-2 rounded-full font-medium transition-all
													${isMuted ? 'bg-gray-500 hover:bg-gray-600' : 'bg-blue-500 hover:bg-blue-600'}`}
											>
												{isMuted ? '🔇 Unmute' : '🔊 Mute'}
											</button>
										</div>
										<div className="flex items-center space-x-2">
											<span className="text-sm">Volume</span>
											<input
												type="range"
												min="0"
												max="100"
												value={volume}
												onChange={(e) => setVolume(parseInt(e.target.value))}
												className="w-24"
											/>
										</div>
									</div>
								</div>
							</div>
						</div>

						{/* Sidebar */}
						<div className="space-y-6">
							{/* Language Selector */}
							<div className="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur-sm">
								<h3 className="text-lg font-semibold mb-3">Sign Language</h3>
								<LanguageSelector
									languages={languages}
									selectedLanguage={selectedLanguage}
									onLanguageChange={handleLanguageChange}
								/>
							</div>

							{/* Current Interpretation */}
							<div className="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur-sm">
								<h3 className="text-lg font-semibold mb-3">Current Interpretation</h3>
								<div className="space-y-2">
									<p className="text-2xl">{gesture} {emoji}</p>
									<div className="flex items-center space-x-2">
										<span className="text-sm opacity-75">Emotion:</span>
										<span>{emotion} ({(emotionConfidence * 100).toFixed(1)}%)</span>
									</div>
								</div>
							</div>

							{/* History */}
							<div className="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur-sm">
								<h3 className="text-lg font-semibold mb-3">Recent History</h3>
								<div className="space-y-2 max-h-48 overflow-y-auto">
									{history.slice().reverse().map((item, index) => (
										<div
											key={index}
											className="px-3 py-2 bg-purple-700 bg-opacity-50 rounded-lg text-sm"
										>
											{item}
										</div>
									))}
								</div>
							</div>
						</div>
					</div>
				)}

				{activeTab === 'video-call' && <VideoCall selectedLanguage={selectedLanguage} />}
				{activeTab === 'profile' && <Profile />}
			</main>
		</div>
	);
};

export default WebcamFeed; 