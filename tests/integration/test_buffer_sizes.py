import sys
import os
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from streaming.wake_controller import WakeController


SAMPLE_RATE = 16000

BUFFER_SIZES = [
    0.25,
    0.5,
    1.0,
    1.5,
    2.0
]


def main():

    print("=== EdgeWake R6 Buffer Size Benchmark ===")

    print(
        "\nSample rate:",
        SAMPLE_RATE,
        "Hz"
    )

    print(
        "Audio format: mono int16 PCM"
    )

    print(
        "\n--------------------------------------------------"
    )

    print(
        f"{'Buffer':>10} | "
        f"{'Samples':>10} | "
        f"{'Memory':>10} | "
        f"{'Released':>10}"
    )

    print(
        "--------------------------------------------------"
    )

    for buffer_seconds in BUFFER_SIZES:

        controller = WakeController(
            buffer_seconds=buffer_seconds,
            sample_rate=SAMPLE_RATE
        )

        # Generate 3 seconds of test audio
        audio = np.arange(
            3 * SAMPLE_RATE,
            dtype=np.int16
        )

        controller.process_audio(audio)

        buffered_audio = (
            controller.handle_kws_result(True)
        )

        samples = len(buffered_audio)

        memory_bytes = (
            samples * np.dtype(np.int16).itemsize
        )

        memory_kb = memory_bytes / 1024

        print(
            f"{buffer_seconds:>8.2f} s | "
            f"{samples:>10} | "
            f"{memory_kb:>8.1f} KB | "
            f"{'YES' if samples > 0 else 'NO':>10}"
        )

    print(
        "--------------------------------------------------"
    )

    print("\n=== VALIDATION ===")

    for buffer_seconds in BUFFER_SIZES:

        controller = WakeController(
            buffer_seconds=buffer_seconds,
            sample_rate=SAMPLE_RATE
        )

        audio = np.arange(
            3 * SAMPLE_RATE,
            dtype=np.int16
        )

        controller.process_audio(audio)

        buffered_audio = (
            controller.handle_kws_result(True)
        )

        expected_samples = int(
            buffer_seconds * SAMPLE_RATE
        )

        if len(buffered_audio) == expected_samples:

            print(
                f"[PASS] {buffer_seconds:.2f}s buffer "
                f"→ {len(buffered_audio)} samples"
            )

        else:

            print(
                f"[FAIL] {buffer_seconds:.2f}s buffer "
                f"→ expected {expected_samples}, "
                f"got {len(buffered_audio)}"
            )


if __name__ == "__main__":
    main()