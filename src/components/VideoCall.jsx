import React, { useEffect } from 'react';

const VideoCall = () => {
  const callId = 'some-call-id'; // Replace with actual callId
  const endCall = () => {
    // Implementation of endCall function
  };

  useEffect(() => {
    // Cleanup function to end call when component unmounts
    return () => {
      if (callId) {
        endCall();
      }
    };
  }, [callId, endCall]);

  return (
    // Rest of the component code
  );
};

export default VideoCall; 