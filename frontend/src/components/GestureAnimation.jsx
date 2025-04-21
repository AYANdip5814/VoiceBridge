import React, { useEffect, useRef, useCallback } from 'react';

const GestureAnimation = ({ landmarks, gesture }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const previousLandmarksRef = useRef(null);

  // Hand landmark connections for visualization
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index finger
    [0, 9], [9, 10], [10, 11], [11, 12], // Middle finger
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring finger
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [0, 5], [5, 9], [9, 13], [13, 17] // Palm connections
  ];

  // Interpolate between previous and current landmarks for smooth animation
  const interpolateLandmarks = useCallback((prev, current, t) => {
    if (!prev || !current || prev.length !== current.length) return current;
    
    return prev.map((p, i) => p + (current[i] - p) * t);
  }, []);

  // Draw the hand landmarks on the canvas
  const drawHand = useCallback((ctx, landmarks, width, height) => {
    if (!landmarks || landmarks.length === 0) return;

    try {
      // Clear the canvas
      ctx.clearRect(0, 0, width, height);
      
      // Set drawing style
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#6c5ce7';
      ctx.fillStyle = '#a29bfe';
      
      // Draw connections
      ctx.beginPath();
      HAND_CONNECTIONS.forEach(([i, j]) => {
        if (i < landmarks.length / 3 && j < landmarks.length / 3) {
          const x1 = landmarks[i * 3] * width;
          const y1 = landmarks[i * 3 + 1] * height;
          const x2 = landmarks[j * 3] * width;
          const y2 = landmarks[j * 3 + 1] * height;
          
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
        }
      });
      ctx.stroke();
      
      // Draw landmarks
      landmarks.forEach((_, i) => {
        if (i % 3 === 0) {
          const x = landmarks[i] * width;
          const y = landmarks[i + 1] * height;
          
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    } catch (error) {
      console.error('Error drawing hand:', error);
    }
  }, [HAND_CONNECTIONS]);

  // Animation loop
  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    try {
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      
      // Interpolate landmarks for smooth animation
      const interpolatedLandmarks = interpolateLandmarks(
        previousLandmarksRef.current,
        landmarks,
        0.3
      );
      
      drawHand(ctx, interpolatedLandmarks, width, height);
      previousLandmarksRef.current = interpolatedLandmarks;
      
      // Add gesture label with shadow for better visibility
      if (gesture) {
        ctx.save();
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        
        // Draw text shadow
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillText(gesture, width / 2 + 1, height - 9);
        
        // Draw text
        ctx.fillStyle = 'white';
        ctx.fillText(gesture, width / 2, height - 10);
        ctx.restore();
      }
      
      animationRef.current = requestAnimationFrame(animate);
    } catch (error) {
      console.error('Error in animation loop:', error);
      cancelAnimationFrame(animationRef.current);
    }
  }, [landmarks, gesture, drawHand, interpolateLandmarks]);

  // Start animation when landmarks change
  useEffect(() => {
    if (!canvasRef.current) return;

    // Ensure canvas is properly sized
    const resizeCanvas = () => {
      const canvas = canvasRef.current;
      if (canvas) {
        const parent = canvas.parentElement;
        if (parent) {
          canvas.width = parent.clientWidth;
          canvas.height = parent.clientHeight;
        }
      }
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    
    if (landmarks && landmarks.length > 0) {
      animate();
    } else {
      // Clear canvas when no landmarks are present
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [landmarks, animate]);

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full bg-indigo-900 bg-opacity-50 rounded-lg"
      />
      <div className="absolute top-2 left-2 bg-indigo-800 bg-opacity-75 text-white px-2 py-1 rounded text-sm">
        Gesture Animation
      </div>
    </div>
  );
};

export default GestureAnimation; 