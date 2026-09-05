from streaming.buffer import AudioRingBuffer


class WakeController:

    def __init__(self, buffer_seconds=1, sample_rate=16000):
        max_samples = int(buffer_seconds * sample_rate)

        self.ring_buffer = AudioRingBuffer(max_samples)

        self.streaming = False

    def process_audio(self, samples):
        """
        Receive continuous microphone PCM audio.

        Audio is always added to the ring buffer.
        If streaming has already started, the current
        audio chunk is returned for transmission.
        """

        self.ring_buffer.add(samples)

        if self.streaming:
            return samples

        return None

    def handle_kws_result(self, detected):
        """
        Handle the output of the KWS detector.

        Parameters
        ----------
        detected : bool
            True when the wake word is detected.

        Returns
        -------
        numpy.ndarray or None
            Buffered audio when wake is detected,
            otherwise None.
        """

        if detected and not self.streaming:
            return self.wake_detected()

        return None

    def wake_detected(self):
        """
        Called when KWS detects the wake word.

        Enables streaming and returns the audio currently
        stored in the ring buffer.
        """

        print("[WAKE] Wake word detected")

        self.streaming = True

        buffered_audio = self.ring_buffer.get_audio()

        print(
            "[WAKE] Buffered samples:",
            len(buffered_audio)
        )

        return buffered_audio

    def stop_streaming(self):
        """
        Stop streaming after the command is complete.
        """

        self.streaming = False

        print("[STREAM] Streaming stopped")