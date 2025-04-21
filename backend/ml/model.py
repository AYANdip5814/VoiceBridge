import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, List, Optional
import numpy as np

def create_isl_model(
    input_shape: Tuple[int, int, int] = (63, 3, 1),
    num_classes: int = 20,  # Default for basic ISL signs
    use_attention: bool = True,
    use_residual: bool = True,
    use_face_features: bool = True
) -> Model:
    """
    Create a CNN model for Indian Sign Language recognition with attention mechanism.
    
    Args:
        input_shape: Shape of input data (num_landmarks, coordinates, channels)
        num_classes: Number of ISL signs to predict
        use_attention: Whether to use attention mechanism
        use_residual: Whether to use residual connections
        use_face_features: Whether to use facial expression features
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Reshape landmarks to 2D
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # Convolutional layers with residual connections
    if use_residual:
        # First conv block
        conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = layers.BatchNormalization()(conv1)
        
        # Residual connection
        if input_shape[1] == 3:  # If input channels match
            residual = x
        else:
            residual = layers.Conv2D(64, (1, 1), padding='same')(x)
        
        x = layers.Add()([conv1, residual])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 1))(x)
        
        # Second conv block
        conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = layers.BatchNormalization()(conv2)
        
        # Residual connection
        residual = layers.Conv2D(128, (1, 1), padding='same')(x)
        x = layers.Add()([conv2, residual])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 1))(x)
        
        # Third conv block
        conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        conv3 = layers.BatchNormalization()(conv3)
        conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = layers.BatchNormalization()(conv3)
        
        # Residual connection
        residual = layers.Conv2D(256, (1, 1), padding='same')(x)
        x = layers.Add()([conv3, residual])
        x = layers.Activation('relu')(x)
    else:
        # Standard convolutional layers
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 1))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 1))(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
    
    # Attention mechanism
    if use_attention:
        # Channel attention
        channel_attention = layers.GlobalAveragePooling2D()(x)
        channel_attention = layers.Dense(256 // 8, activation='relu')(channel_attention)
        channel_attention = layers.Dense(256, activation='sigmoid')(channel_attention)
        channel_attention = layers.Reshape((1, 1, 256))(channel_attention)
        x = layers.Multiply()([x, channel_attention])
        
        # Spatial attention
        spatial_attention = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(x)
        x = layers.Multiply()([x, spatial_attention])
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_isl_sequence_model(
    input_shape: Tuple[int, int] = (30, 63 * 3),
    num_classes: int = 20,  # Default for basic ISL signs
    use_attention: bool = True,
    use_bidirectional: bool = True,
    use_face_features: bool = True
) -> Model:
    """
    Create an LSTM model for sequential Indian Sign Language recognition.
    
    Args:
        input_shape: Shape of input sequence (timesteps, features)
        num_classes: Number of ISL signs to predict
        use_attention: Whether to use attention mechanism
        use_bidirectional: Whether to use bidirectional LSTM
        use_face_features: Whether to use facial expression features
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # LSTM layers
    if use_bidirectional:
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    else:
        x = layers.LSTM(128, return_sequences=True)(inputs)
    
    x = layers.Dropout(0.3)(x)
    
    if use_bidirectional:
        x = layers.Bidirectional(layers.LSTM(64))(x)
    else:
        x = layers.LSTM(64)(x)
    
    x = layers.Dropout(0.3)(x)
    
    # Attention mechanism
    if use_attention:
        # Self-attention
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(64)(attention)
        attention = layers.Permute([2, 1])(attention)
        x = layers.Multiply()([x, attention])
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_isl_transformer_model(
    input_shape: Tuple[int, int] = (30, 63 * 3),
    num_classes: int = 20,  # Default for basic ISL signs
    num_heads: int = 8,
    num_transformer_blocks: int = 4,
    use_face_features: bool = True
) -> Model:
    """
    Create a Transformer model for Indian Sign Language recognition.
    
    Args:
        input_shape: Shape of input sequence (timesteps, features)
        num_classes: Number of ISL signs to predict
        num_heads: Number of attention heads
        num_transformer_blocks: Number of transformer blocks
        use_face_features: Whether to use facial expression features
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Initial dense layer to project input to transformer dimension
    x = layers.Dense(256)(inputs)
    
    # Positional encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    positions = tf.expand_dims(positions, 1)
    positions = tf.tile(positions, [1, 256])
    positions = tf.cast(positions, tf.float32) / 10000.0
    positions = tf.sin(positions)
    x = x + positions
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=256 // num_heads
        )(x, x)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn_output = layers.Dense(1024, activation='relu')(x)
        ffn_output = layers.Dense(256)(ffn_output)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class ISLDataAugmentation:
    """Data augmentation for Indian Sign Language recognition."""
    
    @staticmethod
    def add_noise(landmarks: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
        """Add Gaussian noise to landmarks."""
        noise = np.random.normal(0, noise_factor, landmarks.shape)
        return landmarks + noise
    
    @staticmethod
    def random_rotation(landmarks: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
        """Apply random rotation to landmarks."""
        angle = np.random.uniform(-max_angle, max_angle)
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # Reshape landmarks for rotation
        original_shape = landmarks.shape
        landmarks_reshaped = landmarks.reshape(-1, 3)
        
        # Apply rotation
        rotated_landmarks = np.dot(landmarks_reshaped, rotation_matrix)
        
        # Reshape back to original shape
        return rotated_landmarks.reshape(original_shape)
    
    @staticmethod
    def random_scaling(landmarks: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Apply random scaling to landmarks."""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return landmarks * scale
    
    @staticmethod
    def random_translation(landmarks: np.ndarray, translation_range: float = 0.1) -> np.ndarray:
        """Apply random translation to landmarks."""
        translation = np.random.uniform(-translation_range, translation_range, 3)
        return landmarks + translation
    
    @staticmethod
    def augment(landmarks: np.ndarray, 
                add_noise_prob: float = 0.5,
                rotation_prob: float = 0.5,
                scaling_prob: float = 0.5,
                translation_prob: float = 0.5) -> np.ndarray:
        """Apply random augmentations to landmarks."""
        augmented = landmarks.copy()
        
        if np.random.random() < add_noise_prob:
            augmented = ISLDataAugmentation.add_noise(augmented)
        
        if np.random.random() < rotation_prob:
            augmented = ISLDataAugmentation.random_rotation(augmented)
        
        if np.random.random() < scaling_prob:
            augmented = ISLDataAugmentation.random_scaling(augmented)
        
        if np.random.random() < translation_prob:
            augmented = ISLDataAugmentation.random_translation(augmented)
        
        return augmented 