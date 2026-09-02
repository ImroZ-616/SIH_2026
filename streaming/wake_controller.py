from streaming.buffer import AudioRingBuffer


class WakeController:

    def __init__(self, buffer_seconds=1, sample_rate=16000):
        max_samples = buffer_seconds * sample_rate

        self.ring_buffer = AudioRingBuffer(max_samples)

        self.streaming = False

    def process_audio(self, samples):
        """
        Receive continuous microphone PCM audio.
        """

        self.ring_buffer.add(samples)

        if self.streaming:
            return samples

        return None

    def wake_detected(self):
        """
        Called when KWS detects the wake word.
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
