import React, { useEffect, useState } from 'react';
import {
  Box,
  VStack,
  Text,
  Progress,
  useColorModeValue,
  Badge,
  HStack,
} from '@chakra-ui/react';
import { TranslationResult } from '../types';

interface TranslationDisplayProps {
  translation: TranslationResult | null;
}

export const TranslationDisplay: React.FC<TranslationDisplayProps> = ({ translation }) => {
  const [history, setHistory] = useState<TranslationResult[]>([]);
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  useEffect(() => {
    if (translation) {
      setHistory(prev => [translation, ...prev].slice(0, 10));
    }
  }, [translation]);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'green';
    if (confidence >= 0.6) return 'yellow';
    return 'red';
  };

  return (
    <VStack spacing={4} width="100%" maxW="800px">
      <Box
        width="100%"
        p={4}
        bg={bgColor}
        borderWidth="1px"
        borderColor={borderColor}
        borderRadius="lg"
      >
        <VStack spacing={4} align="stretch">
          <Text fontSize="xl" fontWeight="bold">
            Current Translation
          </Text>
          
          {translation ? (
            <>
              <Text fontSize="lg">{translation.text}</Text>
              <HStack>
                <Badge colorScheme={getConfidenceColor(translation.confidence)}>
                  Confidence: {(translation.confidence * 100).toFixed(1)}%
                </Badge>
                <Text fontSize="sm" color="gray.500">
                  {new Date(translation.timestamp).toLocaleTimeString()}
                </Text>
              </HStack>
              <Progress
                value={translation.confidence * 100}
                colorScheme={getConfidenceColor(translation.confidence)}
                size="sm"
              />
            </>
          ) : (
            <Text color="gray.500">Waiting for translation...</Text>
          )}
        </VStack>
      </Box>

      {history.length > 0 && (
        <Box
          width="100%"
          p={4}
          bg={bgColor}
          borderWidth="1px"
          borderColor={borderColor}
          borderRadius="lg"
        >
          <VStack spacing={4} align="stretch">
            <Text fontSize="xl" fontWeight="bold">
              Translation History
            </Text>
            {history.map((item, index) => (
              <Box key={index} p={2} borderBottomWidth="1px" borderColor={borderColor}>
                <Text>{item.text}</Text>
                <HStack mt={1}>
                  <Badge colorScheme={getConfidenceColor(item.confidence)}>
                    {(item.confidence * 100).toFixed(1)}%
                  </Badge>
                  <Text fontSize="sm" color="gray.500">
                    {new Date(item.timestamp).toLocaleTimeString()}
                  </Text>
                </HStack>
              </Box>
            ))}
          </VStack>
        </Box>
      )}
    </VStack>
  );
}; 