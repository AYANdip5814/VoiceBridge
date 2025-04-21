import os
import tensorflow as tf
from datetime import datetime
from .model import create_isl_model, create_isl_sequence_model, create_isl_transformer_model, ISLDataAugmentation
from .dataset import SignLanguageDataset
from typing import Optional, Tuple, Dict
import logging
import json
import argparse
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomTensorBoardCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir: str, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch = epoch
        
        # Log learning rate
        lr = self.model.optimizer.lr
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.epoch)
        logs['learning_rate'] = lr
        
        # Log model weights histograms
        for layer in self.model.layers:
            if layer.weights:
                for weight in layer.weights:
                    logs[f'{layer.name}/{weight.name}'] = weight
        
        super().on_epoch_end(epoch, logs)

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, log_dir: str):
        super().__init__()
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)
        
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_val_classes = np.argmax(y_val, axis=1)
        
        # Create confusion matrix
        cm = tf.math.confusion_matrix(y_val_classes, y_pred_classes)
        
        # Log confusion matrix
        with self.file_writer.as_default():
            tf.summary.image('confusion_matrix', 
                           tf.expand_dims(tf.expand_dims(cm, 0), -1),
                           step=epoch)

def load_isl_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Load preprocessed ISL data from .npy files.
    
    Args:
        data_path: Path to the directory containing the ISL data files
        
    Returns:
        Tuple of (X_train, y_train, label_map)
    """
    # Load data
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    
    # Load label map
    with open(os.path.join(data_path, 'isl_label_map.json'), 'r') as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    
    return X_train, y_train, label_map

def create_isl_data_generator(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    augment: bool = True
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset with optional ISL-specific augmentation.
    
    Args:
        X: Input data
        y: Labels
        batch_size: Batch size for training
        augment: Whether to apply data augmentation
        
    Returns:
        TensorFlow dataset
    """
    def augment_fn(x, y):
        # Apply random augmentations
        x = ISLDataAugmentation.augment(x)
        return x, y
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if augment:
        dataset = dataset.map(
            lambda x, y: tf.py_function(
                augment_fn,
                [x, y],
                [tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Shuffle and batch
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train_isl_model(
    model_type: str = 'cnn',
    data_path: str = 'data/processed/isl',
    output_dir: str = 'models/isl',
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 0.001,
    use_augmentation: bool = True,
    use_face_features: bool = True,
    model_params: Optional[Dict] = None
) -> None:
    """
    Train an Indian Sign Language recognition model.
    
    Args:
        model_type: Type of model to train ('cnn', 'lstm', or 'transformer')
        data_path: Path to the preprocessed ISL data
        output_dir: Directory to save the trained model
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        use_augmentation: Whether to use data augmentation
        use_face_features: Whether to use facial expression features
        model_params: Additional parameters for model creation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X_train, y_train, label_map = load_isl_data(data_path)
    
    # Split data into train and validation sets
    train_size = int(0.8 * len(X_train))
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    
    # Create data generators
    train_dataset = create_isl_data_generator(
        X_train, y_train, batch_size, augment=use_augmentation
    )
    val_dataset = create_isl_data_generator(
        X_val, y_val, batch_size, augment=False
    )
    
    # Create model
    if model_type == 'cnn':
        model = create_isl_model(
            input_shape=X_train.shape[1:],
            num_classes=len(label_map),
            use_face_features=use_face_features,
            **(model_params or {})
        )
    elif model_type == 'lstm':
        model = create_isl_sequence_model(
            input_shape=X_train.shape[1:],
            num_classes=len(label_map),
            use_face_features=use_face_features,
            **(model_params or {})
        )
    elif model_type == 'transformer':
        model = create_isl_transformer_model(
            input_shape=X_train.shape[1:],
            num_classes=len(label_map),
            use_face_features=use_face_features,
            **(model_params or {})
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create log directory
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', 'isl', model_type, current_time)
    os.makedirs(log_dir, exist_ok=True)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC()]
    )
    
    # Create callbacks
    callbacks = [
        CustomTensorBoardCallback(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        ),
        ConfusionMatrixCallback(
            validation_data=(X_val, y_val),
            log_dir=log_dir
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, f'{model_type}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(os.path.join(output_dir, f'{model_type}_final.h5'))
    
    # Save label map
    with open(os.path.join(output_dir, 'isl_label_map.json'), 'w') as f:
        json.dump(label_map, f)
    
    logger.info("Training completed. Model saved.")
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(val_dataset)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Indian Sign Language recognition model')
    parser.add_argument('--model_type', type=str, default='cnn',
                      choices=['cnn', 'lstm', 'transformer'],
                      help='Type of model to train')
    parser.add_argument('--data_path', type=str, default='data/processed/isl',
                      help='Path to preprocessed ISL data')
    parser.add_argument('--output_dir', type=str, default='models/isl',
                      help='Directory to save trained model')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate for optimization')
    parser.add_argument('--no_augmentation', action='store_true',
                      help='Disable data augmentation')
    parser.add_argument('--no_face_features', action='store_true',
                      help='Disable facial expression features')
    
    args = parser.parse_args()
    
    # Train model
    train_isl_model(
        model_type=args.model_type,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_augmentation=not args.no_augmentation,
        use_face_features=not args.no_face_features
    ) 