"""
SIH 26172 - Audio Loading and Preprocessing Module
Standardizes raw audio waveforms into 16 kHz Mono, 1.0-second float32 arrays
ready for subsequent acoustic feature extraction (MFCC) in Phase 4.
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
import soundfile as sf
import librosa

# Import central audio standards
from config import (
    TARGET_SAMPLE_RATE,
    TARGET_DURATION,
    TARGET_SAMPLES,
    TARGET_CHANNELS,
    AUDIO_NORM_EPSILON,
)


def load_audio(file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file from disk.
    
    Args:
        file_path: Path to the audio file (.wav, etc.)
        
    Returns:
        Tuple of (waveform as np.ndarray, native sample_rate as int)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Read audio using soundfile (fast and robust for WAV files)
    try:
        data, sample_rate = sf.read(str(path), dtype="float32")
    except Exception:
        # Fallback to librosa/audioread if soundfile encounters unsupported format
        data, sample_rate = librosa.load(str(path), sr=None, mono=False)
        data = data.astype(np.float32)

    return data, sample_rate


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Converts multi-channel (stereo) audio to single-channel (mono) by averaging channels.
    
    Args:
        audio: 1D or 2D numpy array
        
    Returns:
        1D float32 numpy array with shape (num_samples,)
    """
    if audio.ndim == 1:
        return audio.astype(np.float32)
    elif audio.ndim == 2:
        # Check if channels are in axis 0 or axis 1
        if audio.shape[0] < audio.shape[1]:
            # Shape is (channels, samples) -> average along axis 0
            mono = np.mean(audio, axis=0)
        else:
            # Shape is (samples, channels) -> average along axis 1
            mono = np.mean(audio, axis=1)
        return mono.astype(np.float32)
    else:
        raise ValueError(f"Unsupported audio shape with {audio.ndim} dimensions: {audio.shape}")


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """
    Resamples audio to target sampling rate (16000 Hz).
    
    Args:
        audio: 1D numpy array
        orig_sr: Original sampling rate in Hz
        target_sr: Target sampling rate in Hz (default: 16000)
        
    Returns:
        Resampled 1D float32 numpy array
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    # Use librosa high-quality polyphase/soxr resampler
    resampled = librosa.resample(
        y=audio,
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type="soxr_hq"
    )
    return resampled.astype(np.float32)


def normalize_audio(
    audio: np.ndarray,
    target_peak: float = 1.0,
    silence_threshold: float = 1e-4
) -> np.ndarray:
    """
    Applies peak normalization to the audio signal.
    
    If the maximum absolute amplitude exceeds `silence_threshold`, scales the signal
    so that the peak absolute amplitude equals `target_peak`.
    If the signal is near-silence, preserves the low amplitude to avoid amplifying background noise.
    
    Args:
        audio: 1D numpy array
        target_peak: Target peak amplitude (default: 1.0)
        silence_threshold: Threshold below which signals are treated as silence
        
    Returns:
        Normalized 1D float32 numpy array
    """
    max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0.0

    if max_val > silence_threshold:
        normalized = audio * (target_peak / (max_val + AUDIO_NORM_EPSILON))
    else:
        # Preserve quiet room / silence audio without amplifying microscopic noise
        normalized = audio.copy()

    # Clip safely to [-1.0, 1.0] to prevent clipping distortion
    clipped = np.clip(normalized, -1.0, 1.0)
    return clipped.astype(np.float32)


def pad_or_trim(
    audio: np.ndarray,
    target_samples: int = TARGET_SAMPLES,
    mode: str = "center"
) -> np.ndarray:
    """
    Pads or trims the audio waveform to exactly `target_samples` (16000 samples = 1.0s).
    
    - If shorter than target_samples: Pads with zeros (center or right padding).
    - If longer than target_samples: Trims to length (center or start cropping).
    
    Args:
        audio: 1D numpy array
        target_samples: Exact output length in samples (default: 16000)
        mode: Alignment mode ('center' or 'start')
        
    Returns:
        1D float32 numpy array of shape (target_samples,)
    """
    n_samples = len(audio)

    if n_samples == target_samples:
        return audio.astype(np.float32)

    if n_samples < target_samples:
        pad_needed = target_samples - n_samples
        if mode == "center":
            pad_left = pad_needed // 2
            pad_right = pad_needed - pad_left
            padded = np.pad(audio, (pad_left, pad_right), mode="constant", constant_values=0.0)
        else:  # right / end padding
            padded = np.pad(audio, (0, pad_needed), mode="constant", constant_values=0.0)
        return padded.astype(np.float32)

    else:  # n_samples > target_samples
        if mode == "center":
            start_idx = (n_samples - target_samples) // 2
            trimmed = audio[start_idx : start_idx + target_samples]
        else:  # start crop
            trimmed = audio[:target_samples]
        return trimmed.astype(np.float32)


def preprocess_audio(
    file_path_or_audio: Union[str, Path, np.ndarray],
    orig_sr: Optional[int] = None,
    target_sr: int = TARGET_SAMPLE_RATE,
    target_samples: int = TARGET_SAMPLES,
    pad_mode: str = "center"
) -> np.ndarray:
    """
    Complete end-to-end audio preprocessing pipeline.
    
    Pipeline Steps:
        1. Load audio (if file path is provided)
        2. Convert to Mono
        3. Resample to target_sr (16 kHz)
        4. Normalize waveform
        5. Pad or trim to target_samples (16,000 samples)
        
    Args:
        file_path_or_audio: File path or raw numpy array
        orig_sr: Original sample rate (required if passing raw array without sample rate)
        target_sr: Target sample rate in Hz (default: 16000)
        target_samples: Target sample count (default: 16000)
        pad_mode: 'center' or 'start'
        
    Returns:
        Standardized 1D float32 numpy array of shape (16000,)
    """
    if isinstance(file_path_or_audio, (str, Path)):
        raw_audio, sr = load_audio(file_path_or_audio)
    elif isinstance(file_path_or_audio, np.ndarray):
        raw_audio = file_path_or_audio
        if orig_sr is None:
            raise ValueError("orig_sr must be specified when passing a raw numpy array.")
        sr = orig_sr
    else:
        raise TypeError(f"Invalid input type: {type(file_path_or_audio)}")

    # 1. Convert to mono
    mono_audio = to_mono(raw_audio)

    # 2. Resample to target sampling rate (16 kHz)
    resampled_audio = resample_audio(mono_audio, orig_sr=sr, target_sr=target_sr)

    # 3. Normalize waveform
    norm_audio = normalize_audio(resampled_audio)

    # 4. Pad or trim to exact sample length (16,000 samples)
    final_audio = pad_or_trim(norm_audio, target_samples=target_samples, mode=pad_mode)

    return final_audio
