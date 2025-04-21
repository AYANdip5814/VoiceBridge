import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  Button,
  VStack,
  HStack,
  Text,
  useToast,
  Select,
  IconButton,
  useColorModeValue,
} from '@chakra-ui/react';
import { FaPlay, FaStop, FaPause, FaVideo } from 'react-icons/fa';
import { CaptureState, VideoSource } from '../types';

interface VideoCaptureProps {
  onStateChange: (state: CaptureState) => void;
}

export const VideoCapture: React.FC<VideoCaptureProps> = ({ onStateChange }) => {
  const [sources, setSources] = useState<VideoSource[]>([]);
  const [selectedSource, setSelectedSource] = useState<string>('');
  const [isCapturing, setIsCapturing] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const toast = useToast();

  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  useEffect(() => {
    loadVideoSources();
    return () => {
      if (isCapturing) {
        stopCapture();
      }
    };
  }, []);

  const loadVideoSources = async () => {
    try {
      const videoSources = await window.electron.getVideoSources();
      setSources(videoSources);
      if (videoSources.length > 0) {
        setSelectedSource(videoSources[0].id);
      }
    } catch (error) {
      toast({
        title: 'Error loading video sources',
        description: 'Failed to get available video sources',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const startCapture = async () => {
    if (!selectedSource) {
      toast({
        title: 'No source selected',
        description: 'Please select a video source',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    try {
      await window.electron.startCapture(selectedSource);
      setIsCapturing(true);
      setIsPaused(false);
      onStateChange({
        isCapturing: true,
        isPaused: false,
        currentSource: sources.find(s => s.id === selectedSource) || null,
      });
    } catch (error) {
      toast({
        title: 'Error starting capture',
        description: 'Failed to start video capture',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const stopCapture = async () => {
    try {
      await window.electron.stopCapture();
      setIsCapturing(false);
      setIsPaused(false);
      onStateChange({
        isCapturing: false,
        isPaused: false,
        currentSource: null,
      });
    } catch (error) {
      toast({
        title: 'Error stopping capture',
        description: 'Failed to stop video capture',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const togglePause = () => {
    setIsPaused(!isPaused);
    onStateChange({
      isCapturing,
      isPaused: !isPaused,
      currentSource: sources.find(s => s.id === selectedSource) || null,
    });
  };

  return (
    <VStack spacing={4} width="100%">
      <Box
        width="100%"
        maxW="800px"
        height="450px"
        bg={bgColor}
        borderWidth="1px"
        borderColor={borderColor}
        borderRadius="lg"
        overflow="hidden"
        position="relative"
      >
        <video
          ref={videoRef}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
          }}
          autoPlay
          muted
        />
      </Box>

      <HStack spacing={4} width="100%" justify="center">
        <Select
          value={selectedSource}
          onChange={(e) => setSelectedSource(e.target.value)}
          width="300px"
          isDisabled={isCapturing}
        >
          {sources.map((source) => (
            <option key={source.id} value={source.id}>
              {source.name}
            </option>
          ))}
        </Select>

        <IconButton
          aria-label="Refresh sources"
          icon={<FaVideo />}
          onClick={loadVideoSources}
          isDisabled={isCapturing}
        />

        {!isCapturing ? (
          <Button
            leftIcon={<FaPlay />}
            colorScheme="green"
            onClick={startCapture}
            isDisabled={!selectedSource}
          >
            Start Capture
          </Button>
        ) : (
          <>
            <IconButton
              aria-label={isPaused ? 'Resume' : 'Pause'}
              icon={isPaused ? <FaPlay /> : <FaPause />}
              onClick={togglePause}
              colorScheme="yellow"
            />
            <IconButton
              aria-label="Stop"
              icon={<FaStop />}
              onClick={stopCapture}
              colorScheme="red"
            />
          </>
        )}
      </HStack>
    </VStack>
  );
}; 