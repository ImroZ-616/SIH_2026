import numpy as np

from audio.mel import create_mel_filterbank


class LogMelFeatureExtractor:
    """
    Convert PCM audio into a Log-Mel spectrogram.

    This is currently a reference implementation for the
    EdgeWake KWS preprocessing pipeline.

    Parameters are configurable because R2 has not yet
    finalized the KWS model configuration.
    """

    def __init__(
        self,
        sample_rate=16000,
        frame_ms=30,
        hop_ms=10,
        n_fft=480,
        n_mels=40,
        epsilon=1e-10
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.epsilon = epsilon

        self.frame_length = int(
            sample_rate * frame_ms / 1000
        )

        self.hop_length = int(
            sample_rate * hop_ms / 1000
        )

        self.window = np.hanning(
            self.frame_length
        )

        self.mel_filterbank = create_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels
        )

    def frame_audio(self, audio):
        """
        Split audio into overlapping frames.
        """

        frames = []

        for start in range(
            0,
            len(audio) - self.frame_length + 1,
            self.hop_length
        ):
            frame = audio[
                start:start + self.frame_length
            ]

            frames.append(frame)

        return np.array(frames)

    def extract(self, audio):
        """
        Convert audio waveform into Log-Mel features.
        """

        audio = np.asarray(
            audio,
            dtype=np.float32
        )

        frames = self.frame_audio(audio)

        if len(frames) == 0:
            return np.empty(
                (0, self.n_mels),
                dtype=np.float32
            )

        log_mel_features = []

        for frame in frames:

            # 1. Apply Hann window
            windowed_frame = (
                frame * self.window
            )

            # 2. FFT
            fft_result = np.fft.rfft(
                windowed_frame,
                n=self.n_fft
            )

            # 3. Power spectrum
            power = (
                np.abs(fft_result) ** 2
            )

            # 4. Mel filter bank
            mel_energy = (
                self.mel_filterbank @ power
            )

            # 5. Log compression
            log_mel = np.log10(
                np.maximum(
                    mel_energy,
                    self.epsilon
                )
            )

            log_mel_features.append(
                log_mel
            )

        return np.asarray(
            log_mel_features,
            dtype=np.float32
        )