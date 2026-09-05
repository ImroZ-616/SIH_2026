from abc import ABC, abstractmethod
import numpy as np


class KWSDetector(ABC):
    """
    Interface for any Keyword Spotting implementation.

    R6 only depends on this interface.
    The actual ML/KWS model can be plugged in later.
    """

    @abstractmethod
    def process(self, audio_chunk: np.ndarray) -> bool:
        """
        Process an incoming audio chunk.

        Parameters
        ----------
        audio_chunk : np.ndarray
            Mono PCM audio.

        Returns
        -------
        bool
            True when the wake word is detected.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset detector state for a new interaction.
        """
        pass