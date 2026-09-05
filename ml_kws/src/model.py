"""
SIH 26172 - R2 ML/KWS Model Architecture Module
Phase 7: Compact 2D CNN definition for Edge / ESP32 Keyword Spotting.

Architecture: Compact-KWS-CNN
- Input Shape: (98, 13, 1) [Time frames, MFCC coefficients, Channels]
- Output Shape: (3,) [Classes: 0=silence, 1=unknown, 2=keyword/ASTRA]
- Total Parameters: ~20,627 (Trainable: 20,435, Non-trainable: 192)
- Quantized INT8 Size: ~21.5 KB
- Target: TensorFlow Lite Micro / ESP32 Deployment
"""

from typing import Optional, Tuple
import keras
from keras import layers


def build_compact_kws_cnn(
    input_shape: Tuple[int, int, int] = (98, 13, 1),
    num_classes: int = 3,
    dropout_rate: float = 0.25,
    name: str = "Compact_KWS_CNN",
) -> keras.Model:
    """Constructs the standard Compact-KWS-CNN model for 3-class keyword spotting.

    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        Shape of input MFCC feature tensor (time_frames, num_mfcc, channels).
        Default is (98, 13, 1).
    num_classes : int
        Number of output classification categories. Default is 3
        (0: silence, 1: unknown, 2: keyword/ASTRA).
    dropout_rate : float
        Dropout probability applied in the classification head for regularization.
        Default is 0.25.
    name : str
        Name of the Keras model. Default is "Compact_KWS_CNN".

    Returns
    -------
    keras.Model
        Instantiated and uncompiled Keras sequential/functional model.
    """
    model = keras.Sequential(
        [
            # Input Layer
            layers.Input(shape=input_shape, name="mfcc_input"),
            
            # Convolutional Block 1 (Preserves spatial resolution, extracts 16 low-level acoustic maps)
            layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                use_bias=False,
                name="conv2d_1",
            ),
            layers.BatchNormalization(name="batch_norm_1"),
            layers.ReLU(name="relu_1"),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding="valid",
                name="max_pooling2d_1",
            ),
            
            # Convolutional Block 2 (Downsamples and expands to 32 mid-level spectro-temporal maps)
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                use_bias=False,
                name="conv2d_2",
            ),
            layers.BatchNormalization(name="batch_norm_2"),
            layers.ReLU(name="relu_2"),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding="valid",
                name="max_pooling2d_2",
            ),
            
            # Convolutional Block 3 (Captures 48 high-level phonetic transitions)
            layers.Conv2D(
                filters=48,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                use_bias=False,
                name="conv2d_3",
            ),
            layers.BatchNormalization(name="batch_norm_3"),
            layers.ReLU(name="relu_3"),
            
            # Classification Head (Global pooling eliminates dense parameter explosion)
            layers.GlobalAveragePooling2D(name="global_average_pooling2d"),
            layers.Dropout(rate=dropout_rate, name="dropout_1"),
            layers.Dense(units=32, activation="relu", name="dense_bottleneck"),
            layers.Dropout(rate=dropout_rate, name="dropout_2"),
            layers.Dense(units=num_classes, activation="softmax", name="dense_output"),
        ],
        name=name,
    )
    return model


if __name__ == "__main__":
    print("=" * 80)
    print("SIH 26172 - ML/KWS Model Definition (Phase 7)")
    print("=" * 80)
    model = build_compact_kws_cnn()
    model.summary()
    total_params = model.count_params()
    trainable_params = sum(p.numpy().size for p in model.trainable_variables)
    non_trainable_params = sum(p.numpy().size for p in model.non_trainable_variables)
    print("=" * 80)
    print(f"Total Parameters         : {total_params:,}")
    print(f"Trainable Parameters     : {trainable_params:,}")
    print(f"Non-Trainable Parameters : {non_trainable_params:,}")
    print(f"Estimated FP32 Size      : {total_params * 4 / 1024:.2f} KB")
    print(f"Estimated INT8 Size      : {total_params * 1 / 1024:.2f} KB")
    print("=" * 80)
