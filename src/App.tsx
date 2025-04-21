import React, { useState } from 'react';
import {
  ChakraProvider,
  Box,
  VStack,
  Heading,
  Container,
  useColorModeValue,
} from '@chakra-ui/react';
import { VideoCapture } from './components/VideoCapture';
import { CaptureState, Settings } from './types';

const App: React.FC = () => {
  const [settings, setSettings] = useState<Settings>({
    language: 'en',
    confidence: 0.8,
    autoStart: false,
    theme: 'light',
  });

  const [captureState, setCaptureState] = useState<CaptureState>({
    isCapturing: false,
    isPaused: false,
    currentSource: null,
  });

  const bgColor = useColorModeValue('gray.50', 'gray.900');
  const textColor = useColorModeValue('gray.800', 'gray.200');

  const handleCaptureStateChange = (newState: CaptureState) => {
    setCaptureState(newState);
  };

  return (
    <ChakraProvider>
      <Box minH="100vh" bg={bgColor} color={textColor}>
        <Container maxW="container.xl" py={8}>
          <VStack spacing={8}>
            <Heading as="h1" size="xl">
              VoiceBridge
            </Heading>
            <Heading as="h2" size="md" color="gray.500">
              Real-time Sign Language Interpreter
            </Heading>
            
            <VideoCapture onCaptureStateChange={handleCaptureStateChange} />
            
            {/* Translation results will be added here */}
            <Box
              w="100%"
              p={4}
              borderRadius="lg"
              bg={useColorModeValue('white', 'gray.800')}
              boxShadow="base"
            >
              <Heading as="h3" size="sm" mb={4}>
                Translation Results
              </Heading>
              {/* Translation component will be added here */}
            </Box>
          </VStack>
        </Container>
      </Box>
    </ChakraProvider>
  );
};

export default App; 