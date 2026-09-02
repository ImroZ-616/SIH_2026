import numpy as np


class AudioRingBuffer:

    def __init__(self, max_samples):
        self.max_samples = max_samples
        self.buffer = np.zeros(max_samples, dtype=np.int16)
        self.write_index = 0
        self.samples_stored = 0

    def add(self, samples):
        samples = np.asarray(samples, dtype=np.int16).flatten()

        for sample in samples:
            self.buffer[self.write_index] = sample

            self.write_index = (
                self.write_index + 1
            ) % self.max_samples

            self.samples_stored = min(
                self.samples_stored + 1,
                self.max_samples
            )

    def get_audio(self):
        if self.samples_stored < self.max_samples:
            return self.buffer[:self.samples_stored].copy()

        return np.concatenate([
            self.buffer[self.write_index:],
            self.buffer[:self.write_index]
        ])

    def clear(self):
        self.buffer.fill(0)
        self.write_index = 0
        self.samples_stored = 0

    def __len__(self):
        return self.samples_stored
