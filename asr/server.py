from faster_whisper import WhisperModel


class ASR:
    def __init__(self):
        print("Loading ASR model...")

        self.model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )

        print("ASR model loaded")

    def transcribe(self, audio_file):
        segments, info = self.model.transcribe(audio_file)

        text = ""

        for segment in segments:
            text += segment.text + " "

        return text.strip()