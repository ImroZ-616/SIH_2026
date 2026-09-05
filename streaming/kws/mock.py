import numpy as np

from streaming.kws.interface import KWSDetector


class MockKWS(KWSDetector):
    """
    Temporary KWS detector for R6 integration testing.

    Simulates wake-word detection after a configurable
    amount of audio has been processed.
    """

    def __init__(
        self,
        sample_rate=16000,
        detection_after_seconds=1.0
    ):
        self.sample_rate = sample_rate
        self.detection_after_samples = int(
            sample_rate * detection_after_seconds
        )

        self.samples_processed = 0
        self.detected = False

    def process(self, audio_chunk: np.ndarray) -> bool:
        """
        Process an audio chunk.

        Returns True only when the simulated wake
        word is detected.
        """

        if self.detected:
            return False

        audio_chunk = np.asarray(
            audio_chunk,
            dtype=np.int16
        ).flatten()

        self.samples_processed += len(audio_chunk)

        if self.samples_processed >= self.detection_after_samples:
            self.detected = True

            print("[MOCK KWS] Wake word detected!")

            return True

        return False

    def reset(self):
        """Reset detector state."""

        self.samples_processed = 0
        self.detected = False