'''
SIH 26172 - Keyword Audio Recording Tool
Interactive CLI tool for recording genuine 'ASTRA' keyword samples
Standard format: 16 kHz, 16-bit PCM, 1.0 second, Mono WAV.
'''

import sys
import time
import wave
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import KEYWORD_DIR

SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
CHANNELS = 1


def record_interactive():
    print("=" * 60)
    print("SIH 26172 - ASTRA Keyword Recording Tool")
    print("=" * 60)
    print(f"Target directory: {KEYWORD_DIR}")
    print(f"Audio specification: {SAMPLE_RATE} Hz, 16-bit PCM, {DURATION} sec, Mono")
    print("-" * 60)

    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("[ERROR] 'sounddevice' or 'numpy' is required for live recording.")
        print("Please ensure the project virtual environment is active.")
        return

    speaker = input("Enter Speaker ID or Name (e.g., spk01, r2, rahul): ").strip()
    if not speaker:
        speaker = "spk01"

    condition = input("Enter condition/distance (e.g., close_mic, 1m_distance, noisy_fan, normal): ").strip()
    if not condition:
        condition = "normal"

    num_samples_str = input("How many samples do you want to record in this batch? [default: 10]: ").strip()
    num_samples = int(num_samples_str) if num_samples_str.isdigit() else 10

    KEYWORD_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = list(KEYWORD_DIR.glob(f"astra_{speaker}_{condition}_*.wav"))
    start_idx = len(existing_files) + 1

    print("\n" + "=" * 60)
    print(f"Ready to record {num_samples} samples for [{speaker}] under [{condition}] condition.")
    print("When prompted, speak the keyword 'ASTRA' clearly within the 1-second window.")
    print("=" * 60 + "\n")

    recorded_count = 0
    for i in range(start_idx, start_idx + num_samples):
        input(f"--> Press [ENTER] to record sample #{i}...")
        print("    Get ready... 3... 2... 1... RECORDING NOW! (Speak 'ASTRA')", flush=True)

        audio_data = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16'
        )
        sd.wait()
        print("    [DONE Recording]")

        filename = f"astra_{speaker}_{condition}_{i:03d}.wav"
        filepath = KEYWORD_DIR / filename

        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        print(f"    Saved: {filename}\n")
        recorded_count += 1

    print("=" * 60)
    print(f"Recording session complete! Recorded {recorded_count} samples.")
    print(f"Total keyword files currently in {KEYWORD_DIR}: {len(list(KEYWORD_DIR.glob('*.wav')))}")
    print("=" * 60)


if __name__ == "__main__":
    record_interactive()
