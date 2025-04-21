import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  Button,
  Text,
  VStack,
  HStack,
  useToast,
  Select,
  IconButton,
  Flex,
  Badge,
  Progress,
  Tooltip,
} from '@chakra-ui/react';
import { FaPlay, FaStop, FaPause, FaSync, FaCamera } from 'react-icons/fa';
import { useTranslation } from 'react-i18next';

interface ISLInterpreterProps {
  onStateChange?: (state: {
    isCapturing: boolean;
    isPaused: boolean;
    selectedSource: string | null;
  }) => void;
}

interface InterpretationResult {
  sign: string;
  confidence: number;
  timestamp: number;
}

const ISLInterpreter: React.FC<ISLInterpreterProps> = ({ onStateChange }) => {
  const { t } = useTranslation();
  const toast = useToast();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoSources, setVideoSources] = useState<string[]>([]);
  const [selectedSource, setSelectedSource] = useState<string | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [interpretations, setInterpretations] = useState<InterpretationResult[]>([]);
  const [currentSign, setCurrentSign] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);

  useEffect(() => {
    loadVideoSources();
    return () => {
      stopCapture();
    };
  }, []);

  useEffect(() => {
    if (onStateChange) {
      onStateChange({
        isCapturing,
        isPaused,
        selectedSource,
      });
    }
  }, [isCapturing, isPaused, selectedSource, onStateChange]);

  const loadVideoSources = async () => {
    try {
      const sources = await window.electron.getVideoSources();
      setVideoSources(sources);
      if (sources.length > 0) {
        setSelectedSource(sources[0]);
      }
    } catch (error) {
      toast({
        title: t('error.loadingSources'),
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const startCapture = async () => {
    if (!selectedSource) {
      toast({
        title: t('error.noSource'),
        description: t('error.selectSource'),
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      return;
    }

    try {
      await window.electron.startCapture(selectedSource);
      setIsCapturing(true);
      setIsPaused(false);
      setInterpretations([]);
      setCurrentSign(null);
      setConfidence(0);
    } catch (error) {
      toast({
        title: t('error.startCapture'),
        description: error.message,
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
    } catch (error) {
      toast({
        title: t('error.stopCapture'),
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const togglePause = () => {
    setIsPaused(!isPaused);
  };

  const handleInterpretation = (result: InterpretationResult) => {
    setInterpretations(prev => [...prev, result].slice(-10)); // Keep last 10 interpretations
    setCurrentSign(result.sign);
    setConfidence(result.confidence);
  };

  return (
    <VStack spacing={4} align="stretch" w="100%">
      <Box
        position="relative"
        w="100%"
        h="400px"
        bg="gray.800"
        borderRadius="lg"
        overflow="hidden"
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
          }}
        />
        <Flex
          position="absolute"
          bottom={0}
          left={0}
          right={0}
          p={4}
          bg="rgba(0, 0, 0, 0.7)"
          justify="space-between"
          align="center"
        >
          <Select
            value={selectedSource || ''}
            onChange={(e) => setSelectedSource(e.target.value)}
            maxW="200px"
            bg="white"
          >
            {videoSources.map((source) => (
              <option key={source} value={source}>
                {source}
              </option>
            ))}
          </Select>
          <HStack spacing={2}>
            <Tooltip label={t('refresh')}>
              <IconButton
                aria-label={t('refresh')}
                icon={<FaSync />}
                onClick={loadVideoSources}
                colorScheme="blue"
              />
            </Tooltip>
            <Tooltip label={t('capture')}>
              <IconButton
                aria-label={t('capture')}
                icon={<FaCamera />}
                onClick={() => window.electron.captureFrame()}
                colorScheme="blue"
              />
            </Tooltip>
            <Tooltip label={isCapturing ? t('stop') : t('start')}>
              <IconButton
                aria-label={isCapturing ? t('stop') : t('start')}
                icon={isCapturing ? <FaStop /> : <FaPlay />}
                onClick={isCapturing ? stopCapture : startCapture}
                colorScheme={isCapturing ? 'red' : 'green'}
              />
            </Tooltip>
            {isCapturing && (
              <Tooltip label={isPaused ? t('resume') : t('pause')}>
                <IconButton
                  aria-label={isPaused ? t('resume') : t('pause')}
                  icon={<FaPause />}
                  onClick={togglePause}
                  colorScheme="yellow"
                />
              </Tooltip>
            )}
          </HStack>
        </Flex>
      </Box>

      <Box p={4} bg="white" borderRadius="lg" shadow="md">
        <VStack spacing={4} align="stretch">
          <HStack justify="space-between">
            <Text fontSize="xl" fontWeight="bold">
              {t('currentSign')}
            </Text>
            <Badge
              colorScheme={confidence > 0.8 ? 'green' : confidence > 0.5 ? 'yellow' : 'red'}
              fontSize="md"
            >
              {Math.round(confidence * 100)}%
            </Badge>
          </HStack>
          <Progress
            value={confidence * 100}
            colorScheme={confidence > 0.8 ? 'green' : confidence > 0.5 ? 'yellow' : 'red'}
            size="sm"
          />
          <Text fontSize="2xl" textAlign="center" fontWeight="bold">
            {currentSign || t('noSignDetected')}
          </Text>
        </VStack>
      </Box>

      <Box p={4} bg="white" borderRadius="lg" shadow="md">
        <Text fontSize="lg" fontWeight="bold" mb={2}>
          {t('recentInterpretations')}
        </Text>
        <VStack spacing={2} align="stretch">
          {interpretations.map((result, index) => (
            <HStack key={index} justify="space-between">
              <Text>{result.sign}</Text>
              <Badge
                colorScheme={result.confidence > 0.8 ? 'green' : result.confidence > 0.5 ? 'yellow' : 'red'}
              >
                {Math.round(result.confidence * 100)}%
              </Badge>
            </HStack>
          ))}
        </VStack>
      </Box>
    </VStack>
  );
};

export default ISLInterpreter; 