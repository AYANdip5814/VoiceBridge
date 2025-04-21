import React, { useEffect, useRef, useState } from 'react';
import { useTheme } from '../context/ThemeContext';

const SignOverlay = ({ gesture, interpretation, landmarks, isVisible, duration = 1.0 }) => {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const startTimeRef = useRef(null);
  const [progress, setProgress] = useState(0);
  const { isDarkMode } = useTheme();

  // Enhanced interpolation with easing and natural movement
  const interpolateLandmarks = (landmarks, progress) => {
    if (!landmarks || landmarks.length === 0) return [];
    
    const result = [];
    
    // Easing functions for smooth transitions
    const easeInOut = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    const easeOutElastic = (t) => {
      const c4 = (2 * Math.PI) / 3;
      return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
    };
    
    // Apply easing to progress
    const t = easeInOut(progress);
    
    // Natural movement patterns
    const wobble = Math.sin(progress * Math.PI * 4) * 0.005;
    const breathe = Math.sin(progress * Math.PI * 2) * 0.01;
    
    for (let i = 0; i < landmarks.length; i += 3) {
      const x = landmarks[i];
      const y = landmarks[i + 1];
      const z = landmarks[i + 2] || 0;
      
      // Apply natural movement and easing
      result.push(
        x + wobble,
        y + breathe,
        z * easeOutElastic(t)
      );
    }
    
    return result;
  };

  // Enhanced hand skeleton drawing with depth and style
  const drawHandSkeleton = (ctx, landmarks, width, height) => {
    if (!landmarks || landmarks.length === 0) return;

    // Define hand connections for a more detailed hand visualization
    const connections = [
      // Thumb
      [0, 1], [1, 2], [2, 3], [3, 4],
      // Index finger
      [0, 5], [5, 6], [6, 7], [7, 8],
      // Middle finger
      [0, 9], [9, 10], [10, 11], [11, 12],
      // Ring finger
      [0, 13], [13, 14], [14, 15], [15, 16],
      // Pinky
      [0, 17], [17, 18], [18, 19], [19, 20],
      // Palm
      [0, 5], [5, 9], [9, 13], [13, 17]
    ];

    // Style configuration
    const style = {
      lineColor: isDarkMode ? 'rgba(255, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.8)',
      jointColor: isDarkMode ? '#ffffff' : '#000000',
      highlightColor: isDarkMode ? '#00ff00' : '#00aa00',
      shadowColor: isDarkMode ? 'rgba(0, 255, 0, 0.2)' : 'rgba(0, 170, 0, 0.2)',
      lineWidth: 2,
      jointRadius: 3,
      shadowBlur: 15
    };

    // Enable shadow for depth effect
    ctx.shadowColor = style.shadowColor;
    ctx.shadowBlur = style.shadowBlur;

    // Draw connections with gradient
    connections.forEach(([i, j]) => {
      const x1 = landmarks[i * 3] * width;
      const y1 = landmarks[i * 3 + 1] * height;
      const z1 = landmarks[i * 3 + 2] || 0;
      
      const x2 = landmarks[j * 3] * width;
      const y2 = landmarks[j * 3 + 1] * height;
      const z2 = landmarks[j * 3 + 2] || 0;

      // Create gradient based on depth
      const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
      const alpha1 = Math.min(1, 0.5 + z1);
      const alpha2 = Math.min(1, 0.5 + z2);
      
      gradient.addColorStop(0, style.lineColor.replace('0.8', alpha1));
      gradient.addColorStop(1, style.lineColor.replace('0.8', alpha2));

      ctx.beginPath();
      ctx.strokeStyle = gradient;
      ctx.lineWidth = style.lineWidth;
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });

    // Draw joints with depth effect
    for (let i = 0; i < landmarks.length; i += 3) {
      const x = landmarks[i] * width;
      const y = landmarks[i + 1] * height;
      const z = landmarks[i + 2] || 0;
      
      // Vary joint size based on depth
      const radius = style.jointRadius * (1 + z * 0.5);
      
      ctx.beginPath();
      ctx.fillStyle = i === 0 ? style.highlightColor : style.jointColor;
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  // Enhanced animation loop with smooth transitions
  const animate = (timestamp) => {
    if (!startTimeRef.current) startTimeRef.current = timestamp;
    const elapsed = timestamp - startTimeRef.current;
    const newProgress = Math.min(elapsed / (duration * 1000), 1);
    
    setProgress(newProgress);
    
    if (newProgress < 1) {
      animationFrameRef.current = requestAnimationFrame(animate);
    }
  };

  useEffect(() => {
    if (!isVisible || !landmarks || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas with fade effect
    ctx.fillStyle = isDarkMode ? 'rgba(0, 0, 0, 0.1)' : 'rgba(255, 255, 255, 0.1)';
    ctx.fillRect(0, 0, width, height);

    // Get interpolated landmarks for current animation frame
    const currentLandmarks = interpolateLandmarks(landmarks, progress);
    
    // Draw hand skeleton
    drawHandSkeleton(ctx, currentLandmarks, width, height);

    // Start animation
    startTimeRef.current = null;
    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [landmarks, isVisible, isDarkMode, progress, duration, animate, drawHandSkeleton]);

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 pointer-events-none z-50">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        width={window.innerWidth}
        height={window.innerHeight}
      />
      <div className={`absolute bottom-8 left-1/2 transform -translate-x-1/2 
        ${isDarkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'} 
        px-6 py-3 rounded-lg shadow-lg transition-all duration-300
        ${progress === 1 ? 'opacity-0 translate-y-4' : 'opacity-100 translate-y-0'}`}>
        <div className="text-center">
          <p className="text-xl font-semibold mb-1">{gesture}</p>
          <p className="text-sm opacity-75">{interpretation}</p>
          <div className="mt-2 h-1 bg-gray-200 rounded-full overflow-hidden">
            <div 
              className="h-full bg-green-500 transition-all duration-100"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignOverlay; 