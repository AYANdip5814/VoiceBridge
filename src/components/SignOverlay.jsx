import React, { useEffect, useRef, useCallback } from 'react';

const SignOverlay = ({ landmarks, isActive }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const lastFrameTimeRef = useRef(0);

  const drawHandSkeleton = useCallback((ctx, landmarks) => {
    if (!landmarks || landmarks.length === 0) return;
    
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    
    // Draw connections between landmarks
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],  // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8],  // Index finger
      [0, 9], [9, 10], [10, 11], [11, 12],  // Middle finger
      [0, 13], [13, 14], [14, 15], [15, 16],  // Ring finger
      [0, 17], [17, 18], [18, 19], [19, 20]  // Pinky
    ];
    
    connections.forEach(([i, j]) => {
      ctx.beginPath();
      ctx.moveTo(landmarks[i].x * ctx.canvas.width, landmarks[i].y * ctx.canvas.height);
      ctx.lineTo(landmarks[j].x * ctx.canvas.width, landmarks[j].y * ctx.canvas.height);
      ctx.stroke();
    });
  }, []);

  const animate = useCallback((timestamp) => {
    if (!lastFrameTimeRef.current) lastFrameTimeRef.current = timestamp;
    const deltaTime = timestamp - lastFrameTimeRef.current;
    
    if (deltaTime >= 16) { // Target 60 FPS
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      if (isActive && landmarks) {
        drawHandSkeleton(ctx, landmarks);
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      
      lastFrameTimeRef.current = timestamp;
    }
    
    animationRef.current = requestAnimationFrame(animate);
  }, [isActive, landmarks, drawHandSkeleton]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Set canvas size to match parent
    const resizeCanvas = () => {
      const parent = canvas.parentElement;
      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;
    };
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Start animation
    animationRef.current = requestAnimationFrame(animate);
    
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [animate]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none'
      }}
    />
  );
};

export default SignOverlay; 